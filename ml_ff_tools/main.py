from enum import Enum

import ase.io as aio
import torch
import typer

from ml_ff_tools.optimizers import minimize_batch

cli = typer.Typer()


class Device(str, Enum):
    cpu = "cpu"
    cuda = "cuda"


@cli.command()
def relax(
    model_path: str,
    data_path: str,
    model_type: str = "mace",
    batch_size: int = 32,
    save_path: str = typer.Option("./relaxed_atoms.xyz", "--save-path", "-s"),
    device: Device = typer.Option(Device.cpu),
    max_num_iters: int = 200,
    max_step_size: float = 1.0,
):
    atoms = aio.read(data_path, index=":", format="extxyz")
    model = torch.load(model_path)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.to(device.value)

    if model_type == "mace":
        from ml_ff_tools.adapters import MACE_Data_Adapter

        adapter = MACE_Data_Adapter(model)
    else:
        raise ValueError("Only mace models supported for now")
    relaxed_atoms = []
    for batch in adapter(atoms, batch_size):
        relaxed_batch = minimize_batch(
            batch, model, adapter, max_step_size=max_step_size, max_steps=max_num_iters
        )
        relaxed_atoms.extend(adapter.batch_to_atoms(relaxed_batch))

    aio.write(save_path, relaxed_atoms, format="extxyz")
