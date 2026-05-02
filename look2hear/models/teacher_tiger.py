from __future__ import annotations

import logging
from collections import OrderedDict

import torch

from .tiger import TIGER

logger = logging.getLogger(__name__)


def _normalize_state_dict_keys(state_dict):
    if any(key.startswith("audio_model.") for key in state_dict):
        return OrderedDict(
            (key[len("audio_model.") :], value)
            for key, value in state_dict.items()
            if key.startswith("audio_model.")
        )
    return state_dict


def _extract_state_dict(payload):
    if isinstance(payload, dict) and "state_dict" in payload:
        return _normalize_state_dict_keys(payload["state_dict"])
    if isinstance(payload, dict):
        return _normalize_state_dict_keys(payload)
    raise TypeError("Unsupported checkpoint payload for teacher TIGER loading.")


def load_teacher_tiger(checkpoint_path: str, model_kwargs: dict, sample_rate: int):
    teacher = TIGER(sample_rate=sample_rate, **model_kwargs)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(payload)
    result = teacher.load_state_dict(state_dict, strict=False)
    # 检查不匹配的键并打印 warning
    if result.missing_keys:
        logger.warning("教师模型缺少的键: %s", result.missing_keys)
    if result.unexpected_keys:
        logger.warning("教师模型多余的键: %s", result.unexpected_keys)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher
