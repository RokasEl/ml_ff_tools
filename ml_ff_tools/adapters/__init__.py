from abc import ABC, abstractmethod
from typing import TypedDict

import torch


class ModelIn(TypedDict):
    positions: torch.Tensor
    batch: torch.Tensor


class ModelOut(TypedDict):
    forces: torch.Tensor
    energy: torch.Tensor


class Model(ABC):
    @abstractmethod
    def __call__(self, *args, **kwds) -> ModelOut:
        pass


from .mace import MACE_Data_Adapter
