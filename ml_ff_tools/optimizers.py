import typing as t
from abc import ABC, abstractmethod

import torch
from mace.tools.scatter import scatter_sum

from ml_ff_tools.adapters import Model, ModelIn

"""
Modifying https://github.com/rfeinman/pytorch-minimize to work with MACE and molecular geometry relaxation
"""


class HessianUpdateStrategy(ABC):
    def __init__(self):
        self.n_updates = 0

    @abstractmethod
    def solve(self, grad):
        pass

    @abstractmethod
    def _update(self, s, y, rho_inv, batch_index):
        pass

    def update(self, s, y, batch_index):
        rho_inv = scatter_sum(s * y, batch_index)
        self._update(s, y, rho_inv, batch_index)
        self.n_updates += 1


class L_BFGS(HessianUpdateStrategy):
    def __init__(
        self, x, batch_size: int, history_size=20, h0_guess: float = 1.0 / 70.0
    ):
        """_summary_

        Args:
            x (_type_): _description_
            history_size (int, optional): _description_. Defaults to 100.
            h0_guess (float, optional): Equivalent to `alpha` in ase.optimize. Defaults to 1./70.
        """
        super().__init__()
        self.y = []
        self.s = []
        self.rho = []
        self.H_diag = torch.ones(batch_size, dtype=x.dtype, device=x.device) * h0_guess
        self.alpha = x.new_empty(history_size).unsqueeze(1).repeat(1, batch_size)
        self.history_size = history_size

    def solve(self, grad: torch.Tensor, batch_index):
        mem_size = len(self.y)
        d = grad.neg()
        for i in reversed(range(mem_size)):
            self.alpha[i] = scatter_sum(self.s[i] * (d), batch_index) * self.rho[i]
            d -= self.y[i] * torch.index_select(self.alpha[i], 0, batch_index)
        d = d * torch.index_select(self.H_diag, 0, batch_index)
        for i in range(mem_size):
            beta_i = scatter_sum(self.y[i] * (d), batch_index) * self.rho[i]
            d += self.s[i] * torch.index_select(self.alpha[i] - beta_i, 0, batch_index)

        return d

    def _update(self, s, y, rho_inv, batch_index):
        if len(self.y) == self.history_size:
            self.y.pop(0)
            self.s.pop(0)
            self.rho.pop(0)
        self.y.append(y)
        self.s.append(s)
        if len(self.rho):
            rho = self.rho[-1].clone()
            rho[rho_inv > 1e-10] = rho_inv[rho_inv > 1e-10].reciprocal()
        else:
            rho = rho_inv.reciprocal()
        self.rho.append(rho)
        self.H_diag[rho_inv > 1e-10] = (
            rho_inv[rho_inv > 1e-10] / scatter_sum(y * y, batch_index)[rho_inv > 1e-10]
        )


supported_methods = t.Literal["lbfgs"]


def minimize_batch(
    batch: ModelIn,
    model: Model,
    adapter,
    f_max: float = 0.02,
    max_step_size: float = 1.0,
    max_steps: int = 200,
    skin_distance: float = 0.1,
    method: supported_methods = "lbfgs",
    **method_kwargs,
):
    x = batch["positions"].detach().view(-1).clone()
    batch_index = batch["batch"].unsqueeze(1).repeat(1, 3).view(-1)
    batch_size = batch_index.max().item() + 1
    out = model(batch)
    force = out["forces"].view(-1)  # negative of the gradient
    gradient = force.neg()
    direction = force

    if method == "lbfgs":
        hess = L_BFGS(x, int(batch_size), **method_kwargs)
    else:
        raise ValueError(
            f"{method} unsupported. Only supported method available: {supported_methods}"
        )

    step_size = scatter_sum(gradient.abs(), batch_index).reciprocal()
    t = torch.minimum(torch.tensor(max_step_size), step_size)
    n_iter = 0
    # Pretty print statement
    print(f"Starting optimization with method {method}")
    print(
        f"{'Step':<10} {'Interaction energy / atom':<40} {'Total Force':<20} {'Max Force':<20}"
    )
    diff = torch.zeros_like(x)
    # BFGS iterations
    for n_iter in range(1, max_steps + 1):
        # ==================================
        #   compute Quasi-Newton direction
        # ==================================

        if n_iter > 1:
            direction = hess.solve(gradient, batch_index)

        x_new = x + direction * torch.index_select(t, 0, batch_index)
        batch["positions"] = x_new.view(-1, 3)
        if (torch.abs(diff) > skin_distance).any():
            print("Updating neighbour list")
            updated_edges, updated_shifts = adapter.update_edge_index(
                batch["positions"], batch["batch"], batch["cell"]
            )
            batch["edge_index"] = updated_edges
            batch["shifts"] = updated_shifts
            diff = torch.zeros_like(x)
        out = model(batch)
        energy = out["interaction_energy"].sum().item() / batch["positions"].shape[0]
        max_per_atom_force = out["forces"].norm(p=2, dim=-1).max().item()
        force_new = out["forces"].view(-1)
        gradient_new = force_new.neg()

        # Hessian update
        s = x_new.sub(x)
        diff += s
        y = gradient_new.sub(gradient)
        hess.update(s, y, batch_index)

        # update state
        x.copy_(x_new)
        gradient.copy_(gradient_new)
        step_size = scatter_sum(gradient.abs(), batch_index).reciprocal()
        t = torch.minimum(torch.tensor(max_step_size), step_size)

        total_grad = gradient.norm(p=2).item()

        # Pretty print statement
        print(
            f"{n_iter:<10} {energy:<40.3f} {total_grad:<20.3e} {max_per_atom_force:<20.3e}"
        )
        if max_per_atom_force <= f_max:
            print(f"Force converged after {n_iter} steps")
            batch["forces"] = out["forces"]
            return batch
    print(
        f"Reached n_iter=max_steps ({max_steps}). Final total force: {gradient.norm(p=2).item():.3e} eV/A"
    )
    batch["forces"] = out["forces"]
    return batch
