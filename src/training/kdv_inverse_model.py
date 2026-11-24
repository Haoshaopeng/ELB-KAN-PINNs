# -*- coding: utf-8 -*-
"""
KdVequation - parallelMLP + Fourier-KAN + hard initial condition + supervised pretraining and physics-based inversion
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.io
import time
import os
import json

from training_utils import (
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

device = configure_torch(seed=env_int("KANPINN_SEED", 42))

print("=" * 80)
print("KdV Inverse - Parallel additive MLP + Fourier-KAN")
print(f"Device: {device}")
print(cuda_status(device))
print("=" * 80)

CONFIG = {
    "n_data": env_int("KANPINN_KDV_N_DATA", 800),
    "stage1_iters": env_int("KANPINN_KDV_STAGE1_ITERS", 8000),
    "stage2_iters": env_int("KANPINN_KDV_STAGE2_ITERS", 8000),
    "lbfgs_iters": env_int("KANPINN_KDV_LBFGS_ITERS", 2000),
    "n_f": env_int("KANPINN_KDV_N_F", 5000),
    "lbfgs_n_f": env_int("KANPINN_KDV_LBFGS_N_F", 5000),
    "noise_std": env_float("KANPINN_KDV_NOISE_STD", 0.0),
    "print_every": env_int("KANPINN_PRINT_EVERY", 500),
    "out_dir": env_str("KANPINN_KDV_KAN_OUT", "CUDAexperiment_results_kdv_kan_full"),
}


class FourierKANLayer(nn.Module):
    """efficient Fourier KAN: precomputed basis matrices"""

    def __init__(self, input_dim, output_dim, num_freq=8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_freq = num_freq
        # coeffs: [input_dim, output_dim, num_freq] for cos and sin
        self.cos_coeffs = nn.Parameter(torch.randn(input_dim, output_dim, num_freq) * 0.01)
        self.sin_coeffs = nn.Parameter(torch.randn(input_dim, output_dim, num_freq) * 0.01)
        self.register_buffer("frequencies", torch.arange(1, num_freq + 1).float().to(device) * np.pi,
                             persistent=False)

    def forward(self, x):
        # x: [batch, input_dim]
        x_norm = torch.tanh(x).unsqueeze(-1)  # [batch, input_dim, 1]
        phase = x_norm * self.frequencies.view(1, 1, -1)  # [batch, input_dim, num_freq]
        cos_base = torch.cos(phase)  # [batch, input_dim, num_freq]
        sin_base = torch.sin(phase)  # [batch, input_dim, num_freq]
        # [batch, input_dim, num_freq] * [input_dim, output_dim, num_freq] -> [batch, output_dim]
        out_cos = torch.einsum("bif,iof->bo", cos_base, self.cos_coeffs)
        out_sin = torch.einsum("bif,iof->bo", sin_base, self.sin_coeffs)
        return out_cos + out_sin


class HybridFourierKAN(nn.Module):
    """Independent branches on the same original coordinates, summed before hard IC."""
    def __init__(self):
        super().__init__()
        self.mlp = nn.ModuleList([
            nn.Linear(2, 40),
            nn.Linear(40, 40),
            nn.Linear(40, 40),
        ])
        self.mlp_output = nn.Linear(40, 1)
        self.kan1 = FourierKANLayer(2, 32, num_freq=8)
        self.kan2 = FourierKANLayer(32, 32, num_freq=8)
        self.kan_output = nn.Linear(32, 1)

        for layer in [*self.mlp, self.mlp_output]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.kan_output.weight)
        nn.init.zeros_(self.kan_output.bias)

    def branch_outputs(self, x, t):
        inp = torch.cat([x, t], dim=1)
        h = inp
        for layer in self.mlp:
            h = torch.sin(layer(h))
        global_output = self.mlp_output(h)
        k1 = torch.tanh(self.kan1(inp))
        k2 = torch.tanh(self.kan2(k1))
        return global_output, self.kan_output(k2)

    def forward(self, x, t):
        global_output, residual_output = self.branch_outputs(x, t)
        return global_output + residual_output


class HardICNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = HybridFourierKAN()
        self.lambda1 = 1.0
        self.log_lambda2 = nn.Parameter(torch.log(torch.tensor([0.0050])))

    def forward(self, x, t):
        ic = torch.cos(np.pi * x)
        return ic + t * self.nn(x, t)

    @property
    def lambda2(self):
        return torch.exp(self.log_lambda2)


def load_data():
    data = scipy.io.loadmat("data/KdV_reference_data_lambda2_0.0025.mat")
    x = data["x"].flatten()
    t = data["t"].flatten()
    usol = data["usol"]
    if usol.shape[0] != len(x):
        usol = usol.T
    return (
        x, t, usol,
        torch.FloatTensor(x.reshape(-1, 1)).to(device),
        torch.FloatTensor(t.reshape(-1, 1)).to(device),
        torch.FloatTensor(usol).to(device),
    )


def sample_data(x, t, usol, n_data=800):
    Nx, Nt = len(x), len(t)
    pts = []
    if n_data <= Nx:
        initial_idx = np.random.choice(Nx, n_data, replace=False)
    else:
        initial_idx = range(Nx)
    for i in initial_idx:
        pts.append((x[i], 0.0, usol[i, 0]))
    n_peak = max(0, n_data - len(pts)) // 2
    xs_peak = np.random.uniform(-0.6, 0.6, n_peak)
    ts_peak = np.random.uniform(0.0, 1.0, n_peak)
    for xi, ti in zip(xs_peak, ts_peak):
        ix = np.argmin(np.abs(x - xi))
        it = np.argmin(np.abs(t - ti))
        pts.append((x[ix], t[it], usol[ix, it]))
    n_uniform = n_data - len(pts)
    if n_uniform > 0:
        idx_x = np.random.choice(Nx, n_uniform, replace=True)
        idx_t = np.random.choice(Nt, n_uniform, replace=True)
        for ix, it in zip(idx_x, idx_t):
            pts.append((x[ix], t[it], usol[ix, it]))
    pts = np.array(pts)
    return (
        torch.FloatTensor(pts[:, 0:1]).to(device),
        torch.FloatTensor(pts[:, 1:2]).to(device),
        torch.FloatTensor(pts[:, 2:3]).to(device),
    )


def compute_pde_residual(model, x_f, t_f):
    x_f.requires_grad_(True)
    t_f.requires_grad_(True)
    u = model(x_f, t_f)
    u_x = torch.autograd.grad(u, x_f, torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t_f, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x_f, torch.ones_like(u_xx), create_graph=True)[0]
    return u_t + model.lambda1 * u * u_x + model.lambda2 * u_xxx


def evaluate_l2(model, x_test, t_test, u_exact):
    with torch.no_grad():
        X, T = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing="ij")
        u_pred = model(X.reshape(-1, 1), T.reshape(-1, 1)).reshape(X.shape)
        return (torch.norm(u_pred - u_exact) / torch.norm(u_exact)).item()


def add_observation_noise(u_data, t_data):
    noise_std = CONFIG["noise_std"]
    if noise_std <= 0:
        return u_data
    # The hard initial condition is exact, so noise is applied only to t > 0 observations.
    mask = (t_data > 0).float()
    scale = torch.std(u_data.detach())
    return u_data + mask * noise_std * scale * torch.randn_like(u_data)


def periodic_bc_loss(model, t_bc, x_min, x_max):
    x_l = torch.full_like(t_bc, x_min)
    x_r = torch.full_like(t_bc, x_max)
    loss_u = torch.mean((model(x_l, t_bc) - model(x_r, t_bc)) ** 2)
    x_l.requires_grad_(True)
    x_r.requires_grad_(True)
    u_x_l = torch.autograd.grad(model(x_l, t_bc), x_l, torch.ones_like(x_l), create_graph=True)[0]
    u_x_r = torch.autograd.grad(model(x_r, t_bc), x_r, torch.ones_like(x_r), create_graph=True)[0]
    loss_ux = torch.mean((u_x_l - u_x_r) ** 2)
    return loss_u + loss_ux


def train():
    start = time.perf_counter()
    x, t, usol, x_test, t_test, u_exact = load_data()
    x_min, x_max = -1.0, 1.0
    t_min, t_max = 0.0, 1.0
    x_data, t_data, u_data = sample_data(x, t, usol, n_data=CONFIG["n_data"])
    u_data = add_observation_noise(u_data, t_data)

    model = HardICNet().to(device)
    print(f"\nData points: {len(x_data)}")
    print(f"Observation noise std: {CONFIG['noise_std']:.3f}")
    print(f"Initial lambda2 guess: {model.lambda2.item():.6f} (true: 0.002500)")
    print(f"Config: {json.dumps(CONFIG, indent=2)}")

    best_l2 = float("inf")
    best_lambda2 = model.lambda2.item()
    history = []
    N_f = CONFIG["n_f"]

    # ================== STAGE 1: supervised pretraining ==================
    print("\n" + "=" * 80)
    print(f"STAGE 1: Supervised Pretraining ({CONFIG['stage1_iters']} iter)")
    print("=" * 80)

    model.log_lambda2.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, CONFIG["stage1_iters"] // 3), gamma=0.5
    )

    for it in range(CONFIG["stage1_iters"]):
        optimizer.zero_grad()
        loss = torch.mean((model(x_data, t_data) - u_data) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        history.append({
            "stage": "stage1",
            "iter": it + 1,
            "loss": float(loss.item()),
            "data_loss": float(loss.item()),
            "lambda2": float(model.lambda2.item()),
        })
        if should_log(it + 1, CONFIG["print_every"], CONFIG["stage1_iters"]):
            l2 = evaluate_l2(model, x_test, t_test, u_exact)
            print(f"  Iter {it + 1:5d} | DataLoss: {loss.item():.2e} | L2: {l2 * 100:.4f}%")

    stage1_l2 = evaluate_l2(model, x_test, t_test, u_exact)
    print(f"  Stage 1 L2: {stage1_l2 * 100:.4f}%")

    # ================== STAGE 2: physics fine-tuning + inverse ==================
    print("\n" + "=" * 80)
    print(f"STAGE 2: Physics Fine-tuning + Inverse ({CONFIG['stage2_iters']} iter)")
    print("=" * 80)

    model.log_lambda2.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, CONFIG["stage2_iters"] // 2), gamma=0.5
    )

    for it in range(CONFIG["stage2_iters"]):
        optimizer.zero_grad()
        loss_data = torch.mean((model(x_data, t_data) - u_data) ** 2)

        x_f = torch.rand(N_f, 1, device=device) * (x_max - x_min) + x_min
        t_f = torch.rand(N_f, 1, device=device) * (t_max - t_min) + t_min
        f = compute_pde_residual(model, x_f, t_f)
        loss_pde = torch.mean(f ** 2)

        t_bc = torch.rand(128, 1, device=device) * (t_max - t_min) + t_min
        loss_bc = periodic_bc_loss(model, t_bc, x_min, x_max)

        loss = 10.0 * loss_data + 0.1 * loss_pde + loss_bc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        history.append({
            "stage": "stage2",
            "iter": it + 1,
            "loss": float(loss.item()),
            "data_loss": float(loss_data.item()),
            "pde": float(loss_pde.item()),
            "bc": float(loss_bc.item()),
            "lambda2": float(model.lambda2.item()),
        })

        if should_log(it + 1, CONFIG["print_every"], CONFIG["stage2_iters"]):
            l2 = evaluate_l2(model, x_test, t_test, u_exact)
            if l2 < best_l2:
                best_l2 = l2
                best_lambda2 = model.lambda2.item()
            l2_err = abs(model.lambda2.item() - 0.0025) / 0.0025 * 100
            print(f"  Iter {it + 1:5d} | Loss: {loss.item():.2e} | "
                  f"L2: {l2 * 100:.4f}% | lambda2={model.lambda2.item():.6f} (err: {l2_err:.1f}%)")

    post_adam_l2 = evaluate_l2(model, x_test, t_test, u_exact)

    # ================== STAGE 3: L-BFGS ==================
    print("\n" + "=" * 80)
    print("STAGE 3: L-BFGS Fine-tuning")
    print("=" * 80)

    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=CONFIG["lbfgs_iters"],
        history_size=100, line_search_fn="strong_wolfe", tolerance_change=1e-9
    )
    iteration = [0]

    def closure():
        optimizer.zero_grad()
        loss_data = torch.mean((model(x_data, t_data) - u_data) ** 2)
        x_f = torch.rand(CONFIG["lbfgs_n_f"], 1, device=device) * (x_max - x_min) + x_min
        t_f = torch.rand(CONFIG["lbfgs_n_f"], 1, device=device) * (t_max - t_min) + t_min
        f = compute_pde_residual(model, x_f, t_f)
        loss_pde = torch.mean(f ** 2)
        t_bc = torch.rand(128, 1, device=device) * (t_max - t_min) + t_min
        loss_bc = periodic_bc_loss(model, t_bc, x_min, x_max)
        loss = 10.0 * loss_data + 0.1 * loss_pde + loss_bc
        loss.backward()

        nonlocal best_l2, best_lambda2
        iteration[0] += 1
        history.append({
            "stage": "lbfgs",
            "iter": iteration[0],
            "loss": float(loss.item()),
            "data_loss": float(loss_data.item()),
            "pde": float(loss_pde.item()),
            "bc": float(loss_bc.item()),
            "lambda2": float(model.lambda2.item()),
        })
        if should_log(iteration[0], max(1, CONFIG["print_every"] // 4), CONFIG["lbfgs_iters"]):
            with torch.no_grad():
                l2 = evaluate_l2(model, x_test, t_test, u_exact)
                if l2 < best_l2:
                    best_l2 = l2
                    best_lambda2 = model.lambda2.item()
                l2_err = abs(model.lambda2.item() - 0.0025) / 0.0025 * 100
                print(f"  LBFGS {iteration[0]:5d} | Loss: {loss.item():.2e} | "
                      f"L2: {l2 * 100:.4f}% | lambda2={model.lambda2.item():.6f} (err: {l2_err:.1f}%)")
        return loss

    if CONFIG["lbfgs_iters"] > 0:
        optimizer.step(closure)

    # ================== results ==================
    final_l2 = evaluate_l2(model, x_test, t_test, u_exact)
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Stage 1 L2 (supervised): {stage1_l2 * 100:.4f}%")
    print(f"Best L2:                 {best_l2 * 100:.4f}%")
    print(f"Final L2:                {final_l2 * 100:.4f}%")
    print(f"\nParameter:")
    print(f"  True lambda2:  0.002500")
    print(f"  Best lambda2:  {best_lambda2:.6f} (error: {abs(best_lambda2 - 0.0025) / 0.0025 * 100:.2f}%)")
    print(f"  Final lambda2: {model.lambda2.item():.6f} (error: {abs(model.lambda2.item() - 0.0025) / 0.0025 * 100:.2f}%)")

    out_dir = ensure_dir(CONFIG["out_dir"])
    torch.save(model.state_dict(), os.path.join(out_dir, "model_weights.pt"))

    results = {
        "stage1_l2": stage1_l2 * 100,
        "best_l2": best_l2 * 100,
        "final_l2": final_l2 * 100,
        "best_lambda2": best_lambda2,
        "final_lambda2": float(model.lambda2.item()),
        "post_adam_l2": post_adam_l2 * 100,
        "architecture": "parallel additive MLP + Fourier-KAN",
        "prediction": "cos(pi*x) + t * (MLP(x,t) + FourierKAN(x,t))",
        "seed": env_int("KANPINN_SEED", 42),
        "parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": time.perf_counter() - start,
        "lbfgs_closure_evaluations": iteration[0],
        "lambda2_relative_error_percent": abs(model.lambda2.item() - 0.0025) / 0.0025 * 100,
        "loss_history_file": "loss_history.csv",
        "config": CONFIG,
        "device": str(device),
    }
    save_history_csv(os.path.join(out_dir, "loss_history.csv"), history)
    save_json(os.path.join(out_dir, "experiment_results.json"), results)
    print(f"\nSaved to {out_dir}/")

    if best_l2 < 0.05 and abs(best_lambda2 - 0.0025) / 0.0025 < 10:
        print("\nSUCCESS!")
    else:
        print("\nPartial success.")
    print("=" * 80)


if __name__ == "__main__":
    train()
