import numpy as np
import zntrack
from ase.build import molecule

from ml_ff_tools.adapters import MACE_Data_Adapter
from ml_ff_tools.optimizers import minimize_batch


def test_update_edge_index_gets_the_same_edge_index_as_mace_default():
    remote = zntrack.from_rev(
        "mace_model", remote="/home/rokas/Programming/MACE-Models", rev="small_spice"
    )
    model = remote.get_model()
    model.eval()
    adapter = MACE_Data_Adapter(model)
    data = zntrack.from_rev(
        "reference_data",
        remote="/home/rokas/Programming/MACE-Models",
        rev="small_spice",
    ).get_atoms()

    for batch in adapter(data, batch_size=7):
        edge_index = batch["edge_index"].detach().cpu().numpy()
        edges = {tuple(x) for x in edge_index.T.tolist()}
        edge_index_from_update, _ = adapter.update_edge_index(
            batch["positions"], batch["batch"]
        )
        edge_index_from_update = edge_index_from_update.detach().cpu().numpy()
        edges_from_update = {tuple(x) for x in edge_index_from_update.T.tolist()}
        assert edges == edges_from_update
