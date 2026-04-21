from __future__ import annotations

from collections import OrderedDict

import torch

from .tiger import TIGER


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
    teacher.load_state_dict(state_dict, strict=False)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher
