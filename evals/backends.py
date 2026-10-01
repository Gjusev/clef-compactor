"""Local inference backend: run the open-weights Clef models without the API.

Wraps ``joint_schema_model.systemone`` from the Hugging Face release
(https://huggingface.co/Cloudflare/clef) in the same ``ask()`` interface the
sync compactor expects, so the whole eval pipeline runs unmodified against
weights loaded on a local GPU.

The stock ``load_release_model`` pins the whole backbone to one device via
``device_map={"": device}``. That fits clef-flash (9B, ~18 GB in fp16) on a
single H200 but not on a single 16 GB T4, so :class:`LocalClefClient`
replicates the loader with ``device_map="auto"`` and places the joint schema
head on the device that holds the backbone's final layers.

Requirements (not package dependencies; installed by the Kaggle kernel):
``torch``, ``transformers>=5.10``, ``accelerate``, ``safetensors``,
``huggingface_hub``, ``pillow``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from clef_compactor.config import Settings
from clef_compactor.models import ClefReply, TokenUsage

__all__ = ["LocalClefClient"]


class LocalClefClient:
    """Drop-in ``ask()`` client backed by locally loaded Clef weights.

    Args:
        settings: Validated settings; ``model`` labels the replies.
        model_path: HF repo id (e.g. ``"Cloudflare/clef-flash"``) or a local
            snapshot directory.
        dtype: Torch dtype name for the backbone, ``"float16"`` for Turing GPUs
            (T4), ``"bfloat16"`` on Ampere or newer.
        device_map: Accelerate device map, ``"auto"`` to shard across GPUs.
        max_memory_gib: Per-GPU memory ceiling used with ``device_map="auto"``.
        max_length: Token bound forwarded to ``encode_record`` (default 16384).
    """

    def __init__(
        self,
        settings: Settings,
        model_path: str,
        *,
        dtype: str = "float16",
        device_map: str = "auto",
        max_memory_gib: int = 15,
        max_length: int = 16384,
    ) -> None:
        import torch  # heavy: imported lazily so import cost stays off the API path

        self.torch = torch
        self.settings = settings
        self.max_length = max_length

        path = Path(model_path)
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(model_path))

        joint_schema_dir = str(path)
        if joint_schema_dir not in sys.path:
            sys.path.insert(0, joint_schema_dir)
        import json

        from joint_schema_model import ClefModel, JointSchemaHead  # type: ignore[import-not-found]
        from safetensors.torch import load_file
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        torch_dtype = getattr(torch, dtype)
        backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
            path,
            dtype=torch_dtype,
            device_map=device_map,
            max_memory={index: f"{max_memory_gib}GiB" for index in range(torch.cuda.device_count())},
        )
        backbone.config.use_cache = False

        head_config = json.loads((path / "joint_head_config.json").read_text())
        head = JointSchemaHead(**head_config)
        head.load_state_dict(load_file(path / "joint_head.safetensors"), strict=True)
        head = head.to(device=self._last_device(backbone), dtype=torch_dtype)

        self.model = ClefModel(backbone, head).eval()
        self.processor = AutoProcessor.from_pretrained(path)

    @staticmethod
    def _last_device(backbone: Any) -> Any:
        """The device holding the backbone's final layers (head must match).

        ``hf_device_map`` values may be ints (``0``, ``1``) or strings
        (``"cuda:1"``); normalise both.
        """
        import torch

        devices = list(getattr(backbone, "hf_device_map", {}).values())
        if not devices:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        indexes = [int(str(value).split(":")[-1]) for value in devices]
        return torch.device("cuda", max(indexes))

    def ask(
        self,
        state: str | dict[str, Any],
        questions: dict[str, dict[str, Any]],
        *,
        images: list[str] | None = None,
        model: str | None = None,
    ) -> ClefReply:
        """Answer one scoring request with a single local forward pass."""
        from joint_schema_model import systemone  # type: ignore[import-not-found]

        if not questions:
            raise ValueError("questions must contain at least one entry")
        request = {
            "model": model or self.settings.model,
            "state": state,
            "questions": questions,
        }
        started = time.perf_counter()
        response = systemone(self.model, self.processor, request, max_length=self.max_length)
        latency_ms = (time.perf_counter() - started) * 1000
        return ClefReply(
            model=str(response["model"]),
            answers=response["answers"],
            usage=TokenUsage.from_api(response.get("usage")),
            request_id="local-weights",
            latency_ms=latency_ms,
        )
