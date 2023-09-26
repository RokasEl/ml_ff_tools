from time import perf_counter

import ase.io as aio
import numpy as np
import zntrack
from ase.build import molecule
from ase.optimize import LBFGS
from mace.calculators import MACECalculator

from ml_ff_tools.adapters import MACE_Data_Adapter
from ml_ff_tools.optimizers import minimize_batch


def main(model_path, data_path):
    calc = MACECalculator(model_path, device="cuda", default_dtype="float32")
    atoms = np.asarray(
        aio.read(data_path, index=":", format="extxyz"),
        dtype=object,
    )

    # pick 10 random configs
    rng = np.random.default_rng(0)
    rng.shuffle(atoms)
    atoms = atoms[:32]

    start_ase = perf_counter()
    relaxed_ase = []
    for mol in atoms:
        x = mol.copy()
        x.calc = calc
        dyn = LBFGS(x, force_consistent=False)
        dyn.run(fmax=0.1)
        relaxed_ase.append(x.copy())
    end_ase = perf_counter()

    adapter = MACE_Data_Adapter(calc.models[0])
    relaxed_parallel = []
    start_parallel = perf_counter()
    for batch in adapter(list(atoms), batch_size=32):
        relaxed_batch = minimize_batch(
            batch,
            calc.models[0],
            adapter,
            max_step_size=0.5,
            max_steps=500,
            f_max=0.1,
        )
        relaxed_parallel.extend(adapter.batch_to_atoms(relaxed_batch))
    end_parallel = perf_counter()

    print(f"10 mols with ase LBFGS took {end_ase-start_ase:.2f} seconds")
    forces = np.concatenate([calc.get_forces(x) for x in relaxed_ase], axis=0)
    forces = np.linalg.norm(forces, ord=2, axis=-1)
    print(
        f"Final median and average force magnitude after LBFGS calculation.\n median:{np.median(forces)}, mean: {forces.mean()}"
    )

    print(f"10 mols with ase LBFGS took {end_parallel-start_parallel:.2f} seconds")
    forces = np.concatenate([calc.get_forces(x) for x in relaxed_parallel], axis=0)
    forces = np.linalg.norm(forces, ord=2, axis=-1)
    print(
        f"Final median and average force magnitude after LBFGS calculation.\n median:{np.median(forces)}, mean: {forces.mean()}"
    )


if __name__ == "__main__":
    main(
        "/home/rokas/Programming/MACE-Models/data/SPICE_Mini_for_Gen_swa.model",
        "./tests/small_spice_fragment_test.xyz",
    )
