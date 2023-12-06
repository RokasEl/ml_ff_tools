
# ML FF Tools

This package provides a CLI for relaxing ASE Atoms objects using an ML force field model.

## Installation

Clone with git then run:
```bash
pip install .
```

## Usage

The package comes with a CLI:

```
ml_ff_tools [OPTIONS]
```

**Arguments**:

- `model_path`: Path to saved PyTorch model file. Should be a MACE model.

- `data_path`: Path to data file containing ASE Atoms objects in extended XYZ format.

**Options**:

- `-s, --save-path`: Path to save relaxed XYZ file. Default `./relaxed_atoms.xyz`.

- `--model-type`: Only `mace` models currently supported.

- `-b, --batch-size`: Batch size for minibatch relaxation. Default 32.

- `--device`: Device to run model on. Either `cpu` or `cuda`. Default `cpu`.

- `--max-num-iters`: Max number of minimization iterations. Default 200.

- `--max-step-size`: Max step size for minimization. Default 1.0.

- `--pbc`: Whether to use periodic boundary conditions. Default `False`.

**Example**:

```
ml_ff_tools model.pth data.xyz -s relaxed.xyz --batch-size 64 --device cuda
```

This will relax the atoms in `data.xyz` using the model `model.pth` on a GPU, with a batch size of 64, saving the relaxed structures to `relaxed.xyz`.
