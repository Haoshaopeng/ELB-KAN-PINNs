# -*- coding: utf-8 -*-
"""
CUDA-optimized Burgers forward experiment.

This is the current best Hybrid MLP+Chebyshev-KAN setup, cleaned up so it can
be run in smoke, quick, or full modes without overwriting the original results.
"""

import json
import os

import numpy as np
import scipy.io
import torch
import torch.nn as nn

from training_utils import (
    Timer,
    clear_cuda,
    configure_torch,
    cuda_status,
    ensure_dir,
    env_float,
    env_int,
    env_str,
    save_history_csv,
    save_json,
    should_log,
)


DEVICE = configure_torch(seed=env_int("KANPINN_SEED", 42))
NU = 0.01 / np.pi


class ChebyshevLayer(nn.Module):
    def __init__(self, in_dim, out_dim, degree=12):
        super().__init__()
        self.degree = degree
        self.coeff = nn.Parameter(torch.randn(in_dim, out_dim, degree) * 0.1)

    def forward(self, x):
        basis = [torch.ones_like(x), x]
        for _ in range(2, self.degree):
            basis.append(2 * x * basis[-1] - basis[-2])
        basis = torch.stack(basis, dim=-1)
        return torch.einsum("nik,iok->no", basis, self.coeff)


class KAN(nn.Module):
    def __init__(self, width=32, degree=12):
        super().__init__()
        self.l1 = ChebyshevLayer(2, width, degree)
        self.l2 = ChebyshevLayer(width, width, degree)
        self.out = nn.Linear(width, 1)

    def forward(self, xt):
        h = torch.tanh(self.l1(xt))
        h = torch.tanh(self.l2(h))
        return self.out(h)


class HybridPINN(nn.Module):
    def __init__(self, mlp_width=64, kan_width=32, degree=12):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, mlp_width),
            nn.Tanh(),
            nn.Linear(mlp_width, mlp_width),
            nn.Tanh(),
            nn.Linear(mlp_width, mlp_width),
            nn.Tanh(),
            nn.Linear(mlp_width, 1),
        )
        self.kan = KAN(width=kan_width, degree=degree)
        nn.init.zeros_(self.kan.out.weight)
        nn.init.zeros_(self.kan.out.bias)

    def forward(self, x, t):
        xt = torch.cat([x, 2.0 * t - 1.0], dim=1)
        return self.mlp(xt) + self.kan(xt)


def initial_condition(x):
    return -torch.sin(np.pi * x)


def sample_points(n_f, n_bc, n_ic, shock_focus, device):
    n_shock = int(n_f * shock_focus)
    n_std = n_f - n_shock

    x_f_std = torch.rand(n_std, 1, device=device) * 2 - 1
    t_f_std = torch.rand(n_std, 1, device=device)
    x_f_shock = torch.rand(n_shock, 1, device=device) * 0.8 - 0.4
    t_f_shock = torch.rand(n_shock, 1, device=device)

    x_ic = torch.rand(n_ic, 1, device=device) * 2 - 1
    t_ic = torch.zeros_like(x_ic)
    t_bc = torch.rand(n_bc, 1, device=device)

    return (
        torch.cat([x_f_std, x_f_shock], dim=0),
        torch.cat([t_f_std, t_f_shock], dim=0),
        x_ic,
        t_ic,
        -torch.ones_like(t_bc),
        torch.ones_like(t_bc),
        t_bc,
    )


def compute_loss(model, data):
    x_f_raw, t_f_raw, x_ic, t_ic, x_bc1, x_bc2, t_bc = data
    x_f = x_f_raw.detach().requires_grad_(True)
    t_f = t_f_raw.detach().requires_grad_(True)

    u = model(x_f, t_f)
    u_x = torch.autograd.grad(u, x_f, torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t_f, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, torch.ones_like(u_x), create_graph=True)[0]

    loss_pde = torch.mean((u_t + u * u_x - NU * u_xx) ** 2)
    loss_ic = torch.mean((model(x_ic, t_ic) - initial_condition(x_ic)) ** 2)
    loss_bc = torch.mean(model(x_bc1, t_bc) ** 2) + torch.mean(model(x_bc2, t_bc) ** 2)
    return loss_pde + 10.0 * loss_ic + 10.0 * loss_bc, loss_pde, loss_ic, loss_bc


def evaluate_l2(model, x, t, u_exact, chunk_size=65536):
    model.eval()
    x_grid, t_grid = torch.meshgrid(x, t, indexing="ij")
    x_flat = x_grid.reshape(-1, 1)
    t_flat = t_grid.reshape(-1, 1)
    preds = []
    with torch.no_grad():
        for start in range(0, x_flat.shape[0], chunk_size):
            end = start + chunk_size
            preds.append(model(x_flat[start:end], t_flat[start:end]))
    u_pred = torch.cat(preds, dim=0).reshape(x_grid.shape)
    l2 = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
    model.train()
    return l2, u_pred


def load_burgers_data():
    data = scipy.io.loadmat("data/Burgers_shock_reference.mat")
    x = torch.tensor(data["x"].flatten(), dtype=torch.float32, device=DEVICE)
    t = torch.tensor(data["t"].flatten(), dtype=torch.float32, device=DEVICE)
    u_exact = torch.tensor(data["usol"], dtype=torch.float32, device=DEVICE)
    if u_exact.shape != (len(x), len(t)):
        u_exact = u_exact.T
    return x, t, u_exact


def train():
    config = {
        "mlp_width": env_int("KANPINN_BURGERS_MLP_WIDTH", 64),
        "kan_width": env_int("KANPINN_BURGERS_KAN_WIDTH", 32),
        "degree": env_int("KANPINN_BURGERS_DEGREE", 12),
        "adam_epochs": env_int("KANPINN_BURGERS_ADAM_EPOCHS", 5000),
        "lbfgs_max_iter": env_int("KANPINN_BURGERS_LBFGS_ITERS", 1000),
        "n_f": env_int("KANPINN_BURGERS_N_F", 10000),
        "n_bc": env_int("KANPINN_BURGERS_N_BC", 400),
        "n_ic": env_int("KANPINN_BURGERS_N_IC", 400),
        "lbfgs_n_f": env_int("KANPINN_BURGERS_LBFGS_N_F", 15000),
        "lbfgs_n_bc": env_int("KANPINN_BURGERS_LBFGS_N_BC", 500),
        "lbfgs_n_ic": env_int("KANPINN_BURGERS_LBFGS_N_IC", 500),
        "shock_focus": env_float("KANPINN_BURGERS_SHOCK_FOCUS", 0.5),
        "resample_interval": env_int("KANPINN_BURGERS_RESAMPLE_INTERVAL", 50),
        "print_every": env_int("KANPINN_PRINT_EVERY", 500),
        "out_dir": env_str("KANPINN_BURGERS_OUT", "results_cuda_burgers"),
    }

    out_dir = ensure_dir(config["out_dir"])
    print("=" * 80)
    print("Burgers CUDA Hybrid MLP+Chebyshev-KAN")
    print(cuda_status(DEVICE))
    print(json.dumps(config, indent=2))
    print("=" * 80)

    x_test, t_test, u_exact = load_burgers_data()
    model = HybridPINN(
        mlp_width=config["mlp_width"],
        kan_width=config["kan_width"],
        degree=config["degree"],
    ).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params}")

    timer = Timer()
    l2_initial, _ = evaluate_l2(model, x_test, t_test, u_exact)
    print(f"Initial L2: {l2_initial.item() * 100:.4f}%")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss = float("inf")
    data = None
    history = []
    for epoch in range(1, config["adam_epochs"] + 1):
        if data is None or (epoch - 1) % config["resample_interval"] == 0:
            data = sample_points(
                config["n_f"],
                config["n_bc"],
                config["n_ic"],
                config["shock_focus"],
                DEVICE,
            )

        optimizer.zero_grad(set_to_none=True)
        loss, loss_pde, loss_ic, loss_bc = compute_loss(model, data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        history.append(
            {
                "stage": "adam",
                "iter": epoch,
                "loss": float(loss.item()),
                "pde": float(loss_pde.item()),
                "ic": float(loss_ic.item()),
                "bc": float(loss_bc.item()),
                "elapsed": float(timer.elapsed),
            }
        )

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), os.path.join(out_dir, "Adambest_weights.pt"))

        if should_log(epoch, config["print_every"], config["adam_epochs"]):
            print(
                f"Adam {epoch:5d}/{config['adam_epochs']} | "
                f"loss={loss.item():.3e} pde={loss_pde.item():.3e} "
                f"ic={loss_ic.item():.3e} bc={loss_bc.item():.3e} "
                f"time={timer.elapsed:.1f}s"
            )

    if config["adam_epochs"] > 0 and os.path.exists(os.path.join(out_dir, "Adambest_weights.pt")):
        model.load_state_dict(
            torch.load(os.path.join(out_dir, "Adambest_weights.pt"), map_location=DEVICE, weights_only=True)
        )

    l2_mid, _ = evaluate_l2(model, x_test, t_test, u_exact)
    print(f"After Adam L2: {l2_mid.item() * 100:.4f}%")

    if config["lbfgs_max_iter"] > 0:
        data = sample_points(
            config["lbfgs_n_f"],
            config["lbfgs_n_bc"],
            config["lbfgs_n_ic"],
            config["shock_focus"],
            DEVICE,
        )
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            max_iter=config["lbfgs_max_iter"],
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            loss, loss_pde, loss_ic, loss_bc = compute_loss(model, data)
            loss.backward()
            history.append(
                {
                    "stage": "lbfgs",
                    "iter": sum(1 for row in history if row["stage"] == "lbfgs") + 1,
                    "loss": float(loss.item()),
                    "pde": float(loss_pde.item()),
                    "ic": float(loss_ic.item()),
                    "bc": float(loss_bc.item()),
                    "elapsed": float(timer.elapsed),
                }
            )
            return loss

        lbfgs.step(closure)

    final_loss, _, _, _ = compute_loss(model, data)
    l2_final, _ = evaluate_l2(model, x_test, t_test, u_exact)
    torch.save(model.state_dict(), os.path.join(out_dir, "final_model_weights.pt"))

    results = {
        "l2_error_percent": float(l2_final.item() * 100),
        "initial_l2_percent": float(l2_initial.item() * 100),
        "mid_l2_percent": float(l2_mid.item() * 100),
        "best_adam_loss": float(best_loss),
        "final_loss": float(final_loss.item()),
        "parameters": params,
        "elapsed_seconds": timer.elapsed,
        "loss_history_file": "loss_history.csv",
        "config": config,
        "device": str(DEVICE),
    }
    save_history_csv(os.path.join(out_dir, "loss_history.csv"), history)
    save_json(os.path.join(out_dir, "experiment_results.json"), results)
    clear_cuda()

    print("=" * 80)
    print(f"Final L2: {results['l2_error_percent']:.4f}%")
    print(f"Saved to {out_dir}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    train()
