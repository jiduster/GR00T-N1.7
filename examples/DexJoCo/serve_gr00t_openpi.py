#!/usr/bin/env python
"""Serve a finetuned GR00T-N1.7 checkpoint over the OpenPI websocket protocol.

DexJoCo evaluation (`dexjoco-openpi-eval`) talks to an OpenPI policy server:
it sends dual-arm images + a 46-D state + a prompt, and expects
`{"actions": array[H, 44]}`. Native `run_gr00t_server.py` uses ZMQ with a
nested GR00T observation schema, so it cannot be used as-is.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import traceback
from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np
import tyro
import websockets

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

LOGGER = logging.getLogger("serve_gr00t_openpi")

VIDEO_KEYS = ("ego", "wrist_left", "wrist_right")
STATE_SLICES = {
    "right_tcp": slice(0, 7),
    "left_tcp": slice(7, 14),
    "right_hand": slice(14, 30),
    "left_hand": slice(30, 46),
}
ACTION_KEYS = ("right_tcp", "right_hand", "left_tcp", "left_hand")
IMAGE_ALIASES = {
    "ego": ("ego", "base", "observation.images.ego", "observation.images.front"),
    "wrist_left": ("wrist_left", "observation.images.wrist_left"),
    "wrist_right": ("wrist_right", "observation.images.wrist_right"),
}


def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_Packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


def _as_hwc_uint8(value: Any, *, key: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim == 4 and int(image.shape[0]) == 1:
        image = image[0]
    if image.ndim == 5 and int(image.shape[0]) == 1 and int(image.shape[1]) == 1:
        image = image[0, 0]
    if image.ndim != 3:
        raise ValueError(f"Image `{key}` must be HWC, got {image.shape}")
    if image.shape[-1] not in (1, 3, 4) and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    elif image.shape[-1] != 3:
        raise ValueError(f"Image `{key}` must have 3 channels, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        upper = 1.0 if float(np.nanmax(image)) <= 1.0 else 255.0
        image = np.clip(image, 0.0, upper)
        if upper == 1.0:
            image = image * 255.0
        image = image.round().astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _pick_image(obs: dict[str, Any], names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if name in obs:
            return _as_hwc_uint8(obs[name], key=name)
    raise KeyError(f"None of the image keys {names} were present in the observation")


def dexjoco_obs_to_gr00t(obs: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenPI/DexJoCo observation into Gr00tPolicy input."""
    video = {
        key: _pick_image(obs, IMAGE_ALIASES[key])[None, None, ...]
        for key in VIDEO_KEYS
    }

    if "state" in obs:
        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
    elif "observation.state" in obs:
        state = np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)
    else:
        raise KeyError("Observation is missing `state`")
    if state.shape[0] < 46:
        raise ValueError(f"Expected a 46-D bimanual state, got {state.shape}")

    gr00t_state = {
        name: state[sl][None, None, :].astype(np.float32, copy=False)
        for name, sl in STATE_SLICES.items()
    }

    prompt = obs.get("prompt") or obs.get("task") or ""
    if isinstance(prompt, (list, tuple)):
        prompt = prompt[0]
    prompt = str(prompt)

    return {
        "video": video,
        "state": gr00t_state,
        "language": {"task": [[prompt]]},
    }


def gr00t_action_to_dexjoco(action: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate GR00T action groups into DexJoCo's 44-D dual-arm layout."""
    pieces = []
    for key in ACTION_KEYS:
        if key not in action:
            raise KeyError(f"GR00T action is missing `{key}`: {sorted(action)}")
        value = np.asarray(action[key], dtype=np.float32)
        if value.ndim == 3:
            value = value[0]
        if value.ndim != 2:
            raise ValueError(f"Action `{key}` must be [H, D], got {value.shape}")
        pieces.append(value)
    horizon = pieces[0].shape[0]
    if any(piece.shape[0] != horizon for piece in pieces):
        shapes = {key: action[key].shape for key in ACTION_KEYS}
        raise ValueError(f"Inconsistent action horizons: {shapes}")
    return np.concatenate(pieces, axis=-1)


@dataclass
class Args:
    model_path: str = (
        "/mnt/ceph2/ckpt/gr00t_n1d7_dexjoco_bimanual_microwave/visual_ft_8gpu/visual_ft_8gpu"
    )
    embodiment_tag: str = "NEW_EMBODIMENT"
    device: str = "cuda:0"
    host: str = "0.0.0.0"
    port: int = 8000
    strict: bool = True


class Gr00tOpenPIPolicy:
    def __init__(self, policy: Gr00tPolicy):
        self.policy = policy

    def metadata(self) -> dict[str, Any]:
        return {
            "policy": "gr00t_n1d7",
            "embodiment_tag": self.policy.embodiment_tag.name,
            "action_horizon": 40,
            "action_dim": 44,
        }

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        gr00t_obs = dexjoco_obs_to_gr00t(obs)
        action_dict, _info = self.policy.get_action(gr00t_obs)
        actions = gr00t_action_to_dexjoco(action_dict)
        return {"actions": actions}


class WebsocketPolicyServer:
    def __init__(self, policy: Gr00tOpenPIPolicy, host: str, port: int):
        self.policy = policy
        self.host = host
        self.port = int(port)

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        async with websockets.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            ping_interval=None,
            ping_timeout=None,
        ):
            LOGGER.info("Serving GR00T OpenPI policy on ws://%s:%d", self.host, self.port)
            await asyncio.Future()

    async def _handler(self, websocket) -> None:
        remote = getattr(websocket, "remote_address", None)
        LOGGER.info("Connection from %s opened", remote)
        packer = _Packer()
        await websocket.send(packer.pack(self.policy.metadata()))

        while True:
            try:
                obs = _unpackb(await websocket.recv())
                infer_start = time.monotonic()
                result = self.policy.infer(obs)
                result["server_timing"] = {
                    "infer_ms": (time.monotonic() - infer_start) * 1000.0
                }
                await websocket.send(packer.pack(result))
            except websockets.ConnectionClosed:
                LOGGER.info("Connection from %s closed", remote)
                break
            except Exception:
                LOGGER.exception("Inference failed")
                await websocket.send(traceback.format_exc())
                await websocket.close(code=1011, reason="Internal server error.")
                break


def main(args: Args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    LOGGER.info("Loading GR00T policy from %s", args.model_path)
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(args.embodiment_tag),
        model_path=args.model_path,
        device=args.device,
        strict=args.strict,
    )
    server = WebsocketPolicyServer(
        Gr00tOpenPIPolicy(policy),
        host=args.host,
        port=args.port,
    )
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
