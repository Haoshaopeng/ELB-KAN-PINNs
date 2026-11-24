"""Separate, preregistered forward-PDE comparison with no interior labels."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn

CODE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('forward_existing', CODE / 'benchmark_training.py')
existing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(existing)
NS_SPEC = importlib.util.spec_from_file_location('forward_ns', CODE / 'navier_stokes_diagnostics.py')
ns = importlib.util.module_from_spec(NS_SPEC)
NS_SPEC.loader.exec_module(ns)
BASE = CODE / 'forward_without_interior_labels'
VARIANTS = ['MLP', 'Chebyshev', 'PhysicalFourier']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def seed(value):
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    np.random.seed(value)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def derivative(value, z):
    return torch.autograd.grad(value, z, torch.ones_like(value), create_graph=True)[0]


class PhysicalInput(nn.Module):
    """Spatial harmonics in physical coordinates; nonperiodic time polynomials."""
    def __init__(self, case, width, terms, scale):
        super().__init__()
        self.case, self.terms = case, terms
        self.coeff = nn.Parameter(torch.randn(2 if case == 'KdV' else 3, width, terms) * scale)

    def forward(self, z):
        freq = torch.arange(1, self.terms // 2 + 1, device=z.device, dtype=z.dtype)
        spatial = z[:, :-1] * (math.pi if self.case == 'KdV' else 1.)
        phase = spatial.unsqueeze(-1) * freq
        spatial_basis = torch.cat([phase.cos(), phase.sin()], -1)
        t = 2*z[:, -1:] - 1
        values = [torch.ones_like(t), t]
        for _ in range(2, self.terms):
            values.append(2*t*values[-1] - values[-2])
        time_basis = torch.stack(values, -1)
        return torch.einsum('nik,iok->no', torch.cat([spatial_basis, time_basis], 1), self.coeff)


def make_mlp(inp, widths, out):
    layers = []
    for width in widths:
        layer = nn.Linear(inp, width)
        nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(layer.bias)
        layers.extend([layer, nn.Tanh()])
        inp = width
    layer = nn.Linear(inp, out)
    nn.init.xavier_normal_(layer.weight)
    nn.init.zeros_(layer.bias)
    return nn.Sequential(*layers, layer)


class ForwardModel(nn.Module):
    def __init__(self, case, variant):
        super().__init__()
        self.case, self.variant = case, variant
        inp, out = (2, 1) if case == 'KdV' else (3, 2)
        widths = ([64]*6 if case == 'KdV' else [82, 82, 81]) if variant == 'MLP' else ([40]*3 if case == 'KdV' else [64]*3)
        self.mlp = make_mlp(inp, widths, out)
        if variant != 'MLP':
            width, terms, scale = (32, 16, .01) if case == 'KdV' else (24, 8, .02)
            first = PhysicalInput(case, width, terms, scale) if variant == 'PhysicalFourier' else existing.BasisLayer(inp, width, terms, 'Chebyshev', scale)
            self.kan = nn.Sequential(first, nn.Tanh(),
                existing.BasisLayer(width, width, terms, 'Fourier' if variant == 'PhysicalFourier' else 'Chebyshev', scale),
                nn.Tanh(), nn.Linear(width, out))
            nn.init.zeros_(self.kan[-1].weight)
            nn.init.zeros_(self.kan[-1].bias)
        self.register_buffer('coefficients', torch.tensor([1., .01]))

    def correction(self, z):
        if self.case == 'KdV':
            q = torch.cat([z[:, :1], 2*z[:, 1:2]-1], 1)
        else:
            q = torch.cat([z[:, :2]/math.pi-1, 2*z[:, 2:3]-1], 1)
        result = self.mlp(q)
        if self.variant != 'MLP':
            result = result + self.kan(z if self.variant == 'PhysicalFourier' else q)
        return result

    def forward(self, z):
        correction = self.correction(z)
        if self.case == 'KdV':
            return torch.cos(math.pi*z[:, :1]) + z[:, 1:2]*correction
        x, y, t = z[:, :1], z[:, 1:2], z[:, 2:3]
        psi0 = .5*x.sin()*y.sin() + .15*(2*x).cos()*y.sin() + .1*x.sin()*(3*y).cos()
        # The mean flow is prescribed by the initial condition, not fitted labels.
        psi = .3*y-.2*x + psi0 + t*correction[:, :1]
        return torch.cat([psi, correction[:, 1:2]], 1)


def kdv_residual(model, z):
    u = model(z)
    d = derivative(u, z)
    xx = derivative(d[:, :1], z)[:, :1]
    xxx = derivative(xx, z)[:, :1]
    return d[:, 1:2] + u*d[:, :1] + .0025*xxx


def boundary(model, points):
    if model.case == 'KdV':
        left = torch.cat([-torch.ones_like(points), points], 1).requires_grad_()
        right = torch.cat([torch.ones_like(points), points], 1).requires_grad_()
        a, b = model(left), model(right)
        parts = []
        for order in range(3):
            parts.append((a-b).square().mean())
            if order < 2:
                a = derivative(a, left)[:, :1]
                b = derivative(b, right)[:, :1]
        return torch.stack(parts).sum()
    parts = []
    for axis in [0, 1]:
        # points are (other spatial coordinate, time).
        zeros = torch.zeros_like(points[:, :1])
        zl = torch.cat([zeros, points], 1) if axis == 0 else torch.cat([points[:, :1], zeros, points[:, 1:]], 1)
        zr = zl.clone()
        zr[:, axis] = 2*math.pi
        zl, zr = zl.requires_grad_(), zr.requires_grad_()
        ul, vl, pl = ns.velocity_fields(model, zl)
        ur, vr, pr = ns.velocity_fields(model, zr)
        # Periodic correction psi prevents spurious change in mean flow.
        psi_l = model.correction(zl)[:, :1]
        psi_r = model.correction(zr)[:, :1]
        parts.extend([(psi_l-psi_r).square().mean(), (ul-ur).square().mean(),
                      (vl-vr).square().mean(), (pl-pr).square().mean(),
                      (derivative(ul, zl)[:, axis:axis+1]-derivative(ur, zr)[:, axis:axis+1]).square().mean(),
                      (derivative(vl, zl)[:, axis:axis+1]-derivative(vr, zr)[:, axis:axis+1]).square().mean()])
    return torch.stack(parts).sum()


def sample(case, count, bc_count, generator):
    dims = 2 if case == 'KdV' else 3
    z = torch.rand(count, dims, device='cuda', generator=generator)
    if case == 'KdV':
        z[:, 0] = 2*z[:, 0]-1
        bc = torch.rand(bc_count, 1, device='cuda', generator=generator)
    else:
        z[:, :2] *= 2*math.pi
        bc = torch.rand(bc_count, 2, device='cuda', generator=generator)
        bc[:, 0] *= 2*math.pi
    return z.requires_grad_(), bc


def loss_parts(model, z, bc):
    if model.case == 'KdV':
        physics = kdv_residual(model, z).square().mean()
    else:
        fu, fv, _ = ns.momentum_residual(model, z)
        physics = .5*(fu.square().mean()+fv.square().mean())
    periodic = boundary(model, bc)
    return physics + 10*periodic, physics, periodic


def evaluate(model, folder, smoke):
    """Reference files are opened only after optimization, for final evaluation."""
    model.double()
    if model.case == 'KdV':
        path = CODE / 'benchmark_audit' / 'KdV_crosschecked_reference.mat'
        ref = loadmat(path)
        x, t, exact = ref['x'].ravel(), ref['t'].ravel(), ref['usol']
        xx, tt = np.meshgrid(x, t, indexing='ij')
        z = np.stack([xx.ravel(), tt.ravel()], 1)
        mask = z[:, 1] > 0
        target = exact.ravel()[mask, None]
    else:
        path = CODE / 'NSdiagnostics' / 'independent_solution_diagnostic_data.npz'
        ref = np.load(path)
        z = ref['coordinates']
        mask = z[:, 2] > 0
        target = ref['velocity'][mask]
    z = z[mask]
    if smoke:
        z, target = z[::128], target[::128]
    predicted = []
    for start in range(0, len(z), 2048):
        batch = torch.tensor(z[start:start+2048], device='cuda', dtype=torch.float64)
        if model.case == 'KdV':
            with torch.no_grad():
                p = model(batch)
        else:
            u, v, _ = ns.velocity_fields(model, batch.requires_grad_())
            p = torch.cat([u, v], 1)
        predicted.append(p.detach().cpu().numpy())
    predicted = np.concatenate(predicted)
    error = float(100*np.linalg.norm(predicted-target)/np.linalg.norm(target))
    np.savez_compressed(folder / 'final_prediction.npz', coordinates=z, reference=target, prediction=predicted)
    rng = torch.Generator(device='cuda').manual_seed(20261011)
    zz, bb = sample(model.case, 64 if smoke else 1024, 32 if smoke else 256, rng)
    zz, bb = zz.double().detach().requires_grad_(), bb.double()
    _, physics, periodic = loss_parts(model, zz, bb)
    return dict(field_L2_percent_excluding_initial_time=error,
                independent_physics_rms=float(physics.detach().sqrt()),
                independent_boundary_loss=float(periodic.detach()),
                reference_sha256=sha(path), internal_label_count=0,
                fixed_coefficients=[1., .0025] if model.case == 'KdV' else [1., .01])


def self_test():
    class TaylorGreen(nn.Module):
        coefficients = torch.tensor([1., .01], device='cuda', dtype=torch.float64)
        def forward(self, z):
            x, y, t = z[:, :1], z[:, 1:2], z[:, 2:3]
            a = (-.02*t).exp()
            return torch.cat([a*x.sin()*y.sin(), a*a*((2*x).cos()+(2*y).cos())/4], 1)
    z = torch.rand(32, 3, device='cuda', dtype=torch.float64, requires_grad=True)
    fu, fv, div = ns.momentum_residual(TaylorGreen(), z)
    analytic_max = float(torch.cat([fu, fv, div], 1).abs().max().detach())
    assert analytic_max < 1e-12
    tests = []
    for case in ['KdV', 'NS_periodic']:
        for variant in VARIANTS:
            seed(42)
            model = ForwardModel(case, variant).cuda().double()
            # Perturb all weights, so tests are not vacuous at zero KAN readout.
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(.001*torch.randn_like(p))
            rng = torch.Generator(device='cuda').manual_seed(72)
            z, bc = sample(case, 16, 8, rng)
            z, bc = z.double().detach().requires_grad_(), bc.double()
            loss, _, _ = loss_parts(model, z, bc)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            initial = z.detach().clone()
            initial[:, -1] = 0
            if case == 'KdV':
                ic_error = float((model(initial)-torch.cos(math.pi*initial[:, :1])).abs().max().detach())
                # Independent chain-rule check through an affine coordinate map.
                q = z.detach().clone().requires_grad_()
                physical = torch.stack([q[:, 0], (q[:, 1]+1)/2], 1)
                val = model(physical)
                d = derivative(val, q)
                xx = derivative(d[:, :1], q)[:, :1]
                xxx = derivative(xx, q)[:, :1]
                transformed = 2*d[:, 1:2]+val*d[:, :1]+.0025*xxx
                direct = kdv_residual(model, physical.detach().requires_grad_())
                chain_error = float((transformed-direct).abs().max().detach())
                assert chain_error < 1e-10
            else:
                initial.requires_grad_()
                u, v, _ = ns.velocity_fields(model, initial)
                x, y = initial[:, :1], initial[:, 1:2]
                u0 = .3+.5*x.sin()*y.cos()+.15*(2*x).cos()*y.cos()-.3*x.sin()*(3*y).sin()
                v0 = .2-.5*x.cos()*y.sin()+.3*(2*x).sin()*y.sin()-.1*x.cos()*(3*y).cos()
                ic_error = float(torch.cat([u-u0, v-v0], 1).abs().max().detach())
                chain_error = None
            assert ic_error < 1e-12
            spatial_period_error = None
            if variant == 'PhysicalFourier':
                first = model.kan[0]
                a = z.detach().clone()
                a[:, 0] = -1 if case == 'KdV' else 0
                b = a.clone()
                b[:, 0] = 1 if case == 'KdV' else 2*math.pi
                spatial_period_error = float((first(a)-first(b)).abs().max().detach())
                assert spatial_period_error < 1e-12
            tests.append(dict(case=case, variant=variant, finite_physical_backward=True,
                              hard_initial_error=ic_error, kdv_chain_error=chain_error,
                              physical_first_layer_period_error=spatial_period_error,
                              parameters=sum(p.numel() for p in model.parameters())))
    return dict(Taylor_Green_residual_max=analytic_max, models=tests)


def train(job, root, smoke):
    folder = root / (job['case']+'_'+job['variant']+'_seed'+str(job['seed']))
    folder.mkdir(exist_ok=True)
    source_sha = sha(Path(__file__))
    result_path = folder / 'experiment_results.json'
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding='utf-8'))
        assert result['code_sha256'] == source_sha and result['smoke'] == smoke
        assert result['weights_sha256'] == sha(folder/'final_weights.pt')
        return result
    seed(job['seed'])
    model = ForwardModel(job['case'], job['variant']).cuda()
    rng = torch.Generator(device='cuda').manual_seed(job['seed']+1000)
    steps, nf, nb, lbmax = ((6, 24, 8, 3) if smoke else
                           (8000, 1024, 128, 400) if job['case'] == 'KdV' else (6000, 384, 64, 400))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, max(1, steps//2), gamma=.25)
    history = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    last = time.perf_counter()
    with (folder/'training_log.txt').open('a', encoding='utf-8', buffering=1) as log:
        for step in range(1, steps+1):
            z, bc = sample(job['case'], nf, nb, rng)
            optimizer.zero_grad(set_to_none=True)
            total, physics, periodic = loss_parts(model, z, bc)
            assert torch.isfinite(total), 'Nonfinite physics loss'
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            if step == 1 or step % 100 == 0 or step == steps:
                item = dict(stage='Adam', step=step, loss=float(total.detach()),
                            physics=float(physics.detach()), boundary=float(periodic.detach()),
                            elapsed_seconds=time.perf_counter()-start)
                history.append(item)
                log.write(json.dumps(item)+'\n')
                print(job, item, flush=True)
                save(root/'progress.json', dict(pid=os.getpid(), job=job, step=step, total_steps=steps,
                     elapsed_seconds=time.perf_counter()-start, status='training', smoke=smoke))
            if not smoke and (step % 1000 == 0 or time.perf_counter()-last > 600):
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                scheduler=scheduler.state_dict(), step=step, sampling_rng=rng.get_state()), folder/'intermediate_checkpoint.pt')
                last = time.perf_counter()
        z, bc = sample(job['case'], nf*2, nb*2, rng)
        optimizer = torch.optim.LBFGS(model.parameters(), lr=.8, max_iter=lbmax, max_eval=lbmax*2,
            history_size=50, line_search_fn='strong_wolfe', tolerance_grad=1e-8, tolerance_change=1e-10)
        calls = 0
        def closure():
            nonlocal calls
            optimizer.zero_grad(set_to_none=True)
            total, physics, periodic = loss_parts(model, z, bc)
            assert torch.isfinite(total)
            total.backward()
            calls += 1
            if calls == 1 or calls % 25 == 0:
                item = dict(stage='L-BFGS', calls=calls, loss=float(total.detach()),
                            physics=float(physics.detach()), boundary=float(periodic.detach()))
                history.append(item)
                log.write(json.dumps(item)+'\n')
                save(root/'progress.json', dict(pid=os.getpid(), job=job, status='L-BFGS', calls=calls, smoke=smoke))
            return total
        assert float(closure().detach()) == float(closure().detach())
        optimizer.step(closure)
    torch.save(model.state_dict(), folder/'final_weights.pt')
    torch.cuda.synchronize()
    training_seconds = time.perf_counter()-start
    peak = torch.cuda.max_memory_allocated()
    # Evaluate the saved checkpoint, never choose by reference error.
    model.load_state_dict(torch.load(folder/'final_weights.pt', map_location='cuda', weights_only=True), strict=True)
    metrics = evaluate(model, folder, smoke)
    result = dict(job=job, smoke=smoke, **metrics, code_sha256=source_sha,
                  weights_sha256=sha(folder/'final_weights.pt'), parameters=sum(p.numel() for p in model.parameters()),
                  training_seconds=training_seconds, peak_cuda_memory_bytes=peak, closure_calls=calls,
                  config=dict(adam_steps=steps, nf=nf, boundary_points=nb, lbfgs_max_iter=lbmax,
                              loss='PDE + 10 periodic BC; hard IC; zero interior labels',
                              checkpoint='final only; reference opened after training'))
    save(folder/'training_history.json', history)
    save(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'CUDA required'
    root = BASE / ('smoke_test' if args.smoke else 'experiments')
    root.mkdir(parents=True, exist_ok=True)
    save(root/'physics_initial_and_boundary_checks.json', self_test())
    if args.self_test:
        return
    jobs = [dict(case=case, variant=variant, seed=s) for s in ([42] if args.smoke else [42, 43, 44])
            for case in ['KdV', 'NS_periodic'] for variant in VARIANTS]
    manifest = dict(jobs=jobs, internal_labels=0, seeds=[42, 43, 44], code_sha256=sha(Path(__file__)),
                    protocol='Known-coefficient forward problems, hard initial conditions, common periodic penalties; independent sampling RNG; no label pretraining or reference-based selection.',
                    interpretation='Distinct from supervised inverse experiments. Three seeds are limited evidence. MLP approximately parameter-matched, not identical architecture. Physical Fourier first layer uses spatial physical harmonics and nonperiodic time Chebyshev basis; hidden layer uses Fourier on bounded hidden features. Full mixed output is softly periodic, not strictly periodic.',
                    environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                                     gpu=torch.cuda.get_device_name(), pid=os.getpid()))
    manifest_path = root/'frozen_experiments.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        assert old['jobs'] == jobs and old['code_sha256'] == manifest['code_sha256']
    else:
        save(manifest_path, manifest)
    results = []
    for job in jobs:
        print('START', job, flush=True)
        result = train(job, root, args.smoke)
        results.append(result)
        save(root/'completed_results.json', results)
        print('DONE', result, flush=True)
        torch.cuda.empty_cache()
    save(root/'training_complete.json', dict(completed=len(results), smoke=args.smoke))
    save(root/'progress.json', dict(status='complete', completed=len(results), smoke=args.smoke))


if __name__ == '__main__':
    main()
