"""Exercise a layerwise model and its heat-equation residual on the CPU."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "training"))
from pde_benchmarks import Model, initial, residual


def main():
    torch.manual_seed(42)
    model = Model(dict(case="Heat2D", regime="base", variant="RBF")).double()
    z = torch.tensor([[0.25, 0.4, 0.0], [0.6, 0.7, 0.5]], dtype=torch.float64,
                     requires_grad=True)
    u = model(z)
    torch.testing.assert_close(u[:1], initial(z[:1], model.case, model.p))
    pde = residual(model, z)
    assert pde.shape == (2, 1) and torch.isfinite(pde).all()
    pde.square().mean().backward()
    grad = model.kan[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0
    print("PASS: heat initial condition, PDE residual, and KAN readout gradient (CPU)")


if __name__ == "__main__":
    main()
