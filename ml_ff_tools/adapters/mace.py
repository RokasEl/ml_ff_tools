from functools import partial
from typing import List

import ase
import numpy as np
import torch
from mace.data.atomic_data import AtomicData, get_data_loader
from mace.data.utils import config_from_atoms
from mace.modules import MACE
from mace.tools import AtomicNumberTable
from torch.utils.data import DataLoader
from torch_nl import compute_neighborlist, compute_neighborlist_n2

from ml_ff_tools.adapters.utils import get_model_dtype


def convert_atoms_to_atomic_data(
    atoms: ase.Atoms | List[ase.Atoms],
    z_table: AtomicNumberTable,
    cutoff: float,
    device: str,
):
    if isinstance(atoms, ase.Atoms):
        atoms = [atoms]
    confs = [config_from_atoms(x) for x in atoms]
    atomic_datas = [
        AtomicData.from_config(conf, z_table, cutoff).to(device) for conf in confs
    ]
    return atomic_datas


def batch_atoms(
    atoms: ase.Atoms | list[ase.Atoms],
    z_table: AtomicNumberTable,
    cutoff: float,
    device: str,
) -> AtomicData:
    atomic_datas = convert_atoms_to_atomic_data(atoms, z_table, cutoff, device)
    return next(
        iter(get_data_loader(atomic_datas, batch_size=len(atomic_datas), shuffle=False))
    )


def batch_to_correct_dtype(batch: AtomicData, dtype: torch.dtype):
    if dtype != torch.get_default_dtype():
        keys = filter(
            lambda x: torch.is_floating_point(batch[x]), batch.keys
        )  # type:ignore
        batch = batch.to(dtype, *keys)
        return batch
    else:
        return batch


def change_indices_to_atomic_numbers(
    indices: np.ndarray, z_table: AtomicNumberTable
) -> np.ndarray:
    to_atomic_numbers_fn = np.vectorize(z_table.index_to_z)
    return to_atomic_numbers_fn(indices)


def get_atoms_from_batch(batch, z_table: AtomicNumberTable) -> List[ase.Atoms]:
    """Convert batch to ase.Atoms"""
    atoms_list = []
    for i in range(len(batch.ptr) - 1):
        indices = np.argmax(
            batch.node_attrs[batch.ptr[i] : batch.ptr[i + 1], :].detach().cpu().numpy(),
            axis=-1,
        )
        numbers = change_indices_to_atomic_numbers(indices=indices, z_table=z_table)
        atoms = ase.Atoms(
            numbers=numbers,
            positions=batch.positions[batch.ptr[i] : batch.ptr[i + 1], :]
            .detach()
            .cpu()
            .numpy(),
            cell=None,
        )
        try:
            atoms.arrays["forces"] = (
                batch.forces[batch.ptr[i] : batch.ptr[i + 1], :].detach().cpu().numpy()
            )
        except AttributeError:
            pass
        atoms_list.append(atoms)
    return atoms_list


class MACE_Data_Adapter:
    def __init__(self, model: MACE):
        self.dtype = get_model_dtype(model)
        self.device = model.r_max.device
        self.z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
        self.cutoff = model.r_max.item()  # type: ignore
        self.to_atomic_data = partial(
            convert_atoms_to_atomic_data,
            z_table=self.z_table,
            cutoff=self.cutoff,
            device=self.device,
        )
        self.x0 = None

    def get_dataloader(
        self, atoms: ase.Atoms | list[ase.Atoms], batch_size
    ) -> DataLoader:
        atomic_data = self.to_atomic_data(atoms)
        dataloader = get_data_loader(
            atomic_data, batch_size=batch_size, shuffle=False, drop_last=False
        )
        return dataloader

    def __call__(self, atoms: ase.Atoms | list[ase.Atoms], batch_size: int):
        dataloader = self.get_dataloader(atoms, batch_size=batch_size)
        for batch in dataloader:
            yield batch_to_correct_dtype(batch, self.dtype)

    def batch_to_atoms(self, batch):
        return get_atoms_from_batch(batch, self.z_table)

    def update_edge_index(self, positions, batch_index):
        num_structs = int(batch_index.max()) + 1
        cell = (
            torch.tensor(3 * [0.0, 0.0, 0.0], dtype=self.dtype, device=self.device)
            .view(3, 3)
            .repeat(num_structs, 1, 1)
        )
        pbc = torch.tensor(
            [False, False, False], dtype=torch.bool, device=self.device
        ).repeat(num_structs, 1)
        edge_index, _, shifts = compute_neighborlist_n2(
            self.cutoff, positions, cell, pbc, batch_index
        )
        return edge_index, shifts
