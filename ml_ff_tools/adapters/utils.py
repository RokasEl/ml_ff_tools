import typing as t

import torch


def get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    dtypes = set()
    for p in model.parameters():
        dtypes.add(p.dtype)
    if torch.float32 in dtypes:
        return torch.float32
    elif torch.float64 in dtypes:
        return torch.float64
    else:
        raise ValueError("Model neither float32 or float64")
