import numpy as np
import zntrack
from ase.build import molecule

from ml_ff_tools.adapters import MACE_Data_Adapter
from ml_ff_tools.optimizers import minimize_batch


def test_after_lbfgs_optimization_forces_are_small():
    remote = zntrack.from_rev(
        "mace_model", remote="/home/rokas/Programming/MACE-Models", rev="small_spice"
    )
    model = remote.get_model()
    model.eval()
    adapter = MACE_Data_Adapter(model)
    data = (
        [molecule("C6H6") for _ in range(16)]
        + [molecule("CH3CH2OH") for _ in range(16)]
        + [molecule("CH4") for _ in range(16)]
    )
    print(data)
    rng = np.random.default_rng(2023)
    for atoms in data:
        atoms.positions += rng.normal(size=atoms.positions.shape) * 0.1
    for batch in adapter(data, batch_size=18):
        relaxed_batch = minimize_batch(
            batch, model, adapter, max_steps=200, max_step_size=1
        )
        forces = model(relaxed_batch)["forces"].detach().cpu().numpy()
        np.testing.assert_array_less(forces, 0.02)
