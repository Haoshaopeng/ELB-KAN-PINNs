"""Independent NS references and prior-free, matched-budget diagnostic runs.

Original demonstration-data experiments are never overwritten. Both diagnostic
models predict streamfunction and pressure, with velocity observations and the
physical-coordinate momentum equations used in the original PINNs benchmark.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import scipy.io
from scipy.integrate import solve_ivp
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent / 'NSdiagnostics'
ROOT.mkdir(exist_ok=True)
SEED = 42
TRUTH = np.array([1.0, 0.01])
LOCAL_WAKE = Path(os.environ.get('ELB_WAKE_FILE', ROOT / 'cylinder_nektar_wake.mat'))


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'NS-reproducibility-audit'})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def central4(a, axis, h, order):
    if order == 1:
        return (np.roll(a, 2, axis) - 8*np.roll(a, 1, axis)
                + 8*np.roll(a, -1, axis) - np.roll(a, -2, axis)) / (12*h)
    return (-np.roll(a, 2, axis) + 16*np.roll(a, 1, axis) - 30*a
            + 16*np.roll(a, -1, axis) - np.roll(a, -2, axis)) / (12*h*h)


def verify_wake(data):
    xy, times = data['X_star'], data['t'].ravel()
    xs, ys = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    assert len(xs)*len(ys) == len(xy)
    ix, iy = np.searchsorted(xs, xy[:, 0]), np.searchsorted(ys, xy[:, 1])
    fields = np.empty((len(xs), len(ys), len(times), 3))
    fields[ix, iy, :, :2] = data['U_star'].transpose(0, 2, 1)
    fields[ix, iy, :, 2] = data['p_star']
    dx, dy, dt = np.diff(xs).mean(), np.diff(ys).mean(), np.diff(times).mean()
    assert np.allclose(np.diff(xs), dx) and np.allclose(np.diff(ys), dy)
    u, v, p = (fields[..., j] for j in range(3))
    ux, uy = central4(u, 0, dx, 1), central4(u, 1, dy, 1)
    vx, vy = central4(v, 0, dx, 1), central4(v, 1, dy, 1)
    ut, vt = central4(u, 2, dt, 1), central4(v, 2, dt, 1)
    px, py = central4(p, 0, dx, 1), central4(p, 1, dy, 1)
    lu = central4(u, 0, dx, 2) + central4(u, 1, dy, 2)
    lv = central4(v, 0, dx, 2) + central4(v, 1, dy, 2)
    au, av = u*ux + v*uy, u*vx + v*vy
    interior = (slice(4, -4), slice(4, -4), slice(2, -2))
    lhs = np.concatenate([(ut+px)[interior].ravel(), (vt+py)[interior].ravel()])
    adv = np.concatenate([au[interior].ravel(), av[interior].ravel()])
    lap = np.concatenate([lu[interior].ravel(), lv[interior].ravel()])
    design = np.column_stack([-adv, lap])
    fit = np.linalg.lstsq(design, lhs, rcond=None)[0]
    residual = lhs - design @ TRUTH
    relative = float(np.linalg.norm(residual) / np.linalg.norm(lhs))
    audit = {'grid_shape': list(fields.shape), 'fd_stencil': 'fourth-order centered, boundary excluded',
             'interior_points': len(lhs)//2, 'fd_fitted_lambda': fit.tolist(),
             'reference_momentum_relative_residual': relative,
             'reference_momentum_rms_residual': float(np.sqrt(np.mean(residual**2))),
             'divergence_rms': float(np.sqrt(np.mean((ux+vy)[interior]**2))),
             'coefficient_design_condition': float(np.linalg.cond(design)),
             'finite_difference_verification_not_exact_solver_residual': True}
    # This verifies consistency to finite-difference resolution, not exact zero.
    assert relative < 0.05, audit
    assert np.all(np.abs(fit/TRUTH-1) < 0.1), audit
    return audit


def prepare_wake():
    print('Verifying official cylinder-wake bytes...', flush=True)
    commit = json.loads(download('https://api.github.com/repos/maziarraissi/PINNs/commits/master'))['sha']
    url = f'https://raw.githubusercontent.com/maziarraissi/PINNs/{commit}/main/Data/cylinder_nektar_wake.mat'
    metadata = json.loads(download(f'https://api.github.com/repos/maziarraissi/PINNs/contents/main/Data/cylinder_nektar_wake.mat?ref={commit}'))
    official_bytes = LOCAL_WAKE.read_bytes()
    blob_hash = hashlib.sha1(f'blob {len(official_bytes)}\0'.encode()+official_bytes).hexdigest()
    assert metadata['sha'] == blob_hash and metadata['size'] == len(official_bytes), 'Local bytes differ from official Git blob'
    official_hash = hashlib.sha256(official_bytes).hexdigest()
    assert official_hash == sha(LOCAL_WAKE), 'Local file differs from official immutable reference'
    path = ROOT / 'cylinder_wake_upstream_data.mat'
    if path.exists():
        assert sha(path) == official_hash
    else:
        path.write_bytes(official_bytes)
    data = scipy.io.loadmat(path)
    audit = verify_wake(data)
    source = {'source_url': url, 'repository_commit': commit, 'sha256': official_hash,
              'reference_source_type': 'Public Nektar spectral/hp-element CFD data, not laboratory measurements',
              'primary_paper_url': 'https://arxiv.org/pdf/1711.10566',
              'official_git_blob_sha1': metadata['sha'], 'local_git_blob_sha1': blob_hash,
              'byte_verification_method': 'Git blob hash and size matched against immutable official GitHub contents API',
              'local_source': str(LOCAL_WAKE), 'local_matches_official_bytes': True,
              'original_code_url': f'https://github.com/maziarraissi/PINNs/blob/{commit}/main/continuous_time_identification%20(Navier-Stokes)/NavierStokes.py',
              'reference_lambda': TRUTH.tolist(), 'verification': audit}
    save_json(ROOT / 'cylinder_wake_source_and_equation_checks.json', source)
    print(json.dumps(audit, indent=2), flush=True)
    # Predeclared diagnostic window: all spatial points, first five time units.
    keep = data['t'].ravel() <= 5.0
    t = data['t'].ravel()[keep]
    xy = data['X_star']
    coords = np.column_stack([np.repeat(xy[:, 0], len(t)), np.repeat(xy[:, 1], len(t)), np.tile(t, len(xy))])
    velocity = data['U_star'][:, :, keep].transpose(0, 2, 1).reshape(-1, 2)
    pressure = data['p_star'][:, keep].reshape(-1, 1)
    np.savez_compressed(ROOT / 'cylinder_wake_diagnostic_data.npz', coordinates=coords, velocity=velocity,
                        pressure=pressure, truth=TRUTH, bounds=np.array([[1., -2., 0.], [8., 2., 5.]]))
    save_json(ROOT / 'cylinder_wake_diagnostic_window.json', {'time_window': [0., 5.], 'original_time_window': [0., 19.9],
              'scope': 'Five-time-unit diagnostic, not a reproduction of the full-time original experiment',
              'observed_channels': ['u', 'v'], 'pressure_labels_used_in_training': False})


def spectral_solution(n, times, rtol, atol):
    axis = np.arange(n)*2*np.pi/n
    x, y = np.meshgrid(axis, axis, indexing='ij')
    k = np.fft.fftfreq(n, 1/n)
    kx, ky = np.meshgrid(k, k, indexing='ij')
    k2 = kx*kx + ky*ky
    inverse = np.divide(1, k2, out=np.zeros_like(k2), where=k2 != 0)
    # Strict 2/3 truncation prevents aliasing of all retained quadratic modes.
    mask = (np.abs(kx) < n/3) & (np.abs(ky) < n/3)
    psi = .5*np.sin(x)*np.sin(y) + .15*np.cos(2*x)*np.sin(y) + .1*np.sin(x)*np.cos(3*y)
    w0 = np.fft.fft2(psi)*k2*mask
    mean_u, mean_v = .3, .2

    def terms(wh):
        ph = wh*inverse
        u = mean_u + np.fft.ifft2(1j*ky*ph).real
        v = mean_v - np.fft.ifft2(1j*kx*ph).real
        wx = np.fft.ifft2(1j*kx*wh).real
        wy = np.fft.ifft2(1j*ky*wh).real
        advhat = np.fft.fft2(u*wx+v*wy)*mask
        return u, v, advhat

    def rhs(_, flat):
        wh = flat.reshape(n, n)*mask
        _, _, advhat = terms(wh)
        return (-TRUTH[0]*advhat-TRUTH[1]*k2*wh).ravel()

    start = time.perf_counter()
    solution = solve_ivp(rhs, (times[0], times[-1]), w0.ravel(), method='DOP853',
                         t_eval=times, rtol=rtol, atol=atol, max_step=.025)
    assert solution.success, solution.message
    velocity, pressure, vorticity, energy, enstrophy = [], [], [], [], []
    advection, diffusion, time_derivative = [], [], []
    divergence, momentum_rms = [], []
    for i in range(len(times)):
        wh = solution.y[:, i].reshape(n, n)*mask
        ph = wh*inverse
        u, v, advhat = terms(wh)
        ux = np.fft.ifft2(-kx*ky*ph).real
        uy = np.fft.ifft2(-ky*ky*ph).real
        vx = np.fft.ifft2(kx*kx*ph).real
        vy = np.fft.ifft2(kx*ky*ph).real
        pressure_hat = TRUTH[0]*np.fft.fft2(ux*ux + 2*uy*vx + vy*vy)*inverse*mask
        p = np.fft.ifft2(pressure_hat).real
        wt = rhs(times[i], wh.ravel()).reshape(n, n)
        ut = np.fft.ifft2(1j*ky*wt*inverse).real
        vt = -np.fft.ifft2(1j*kx*wt*inverse).real
        lapu = np.fft.ifft2(-k2*1j*ky*ph).real
        lapv = -np.fft.ifft2(-k2*1j*kx*ph).real
        pu = np.fft.ifft2(1j*kx*pressure_hat).real
        pv = np.fft.ifft2(1j*ky*pressure_hat).real
        ru = ut+TRUTH[0]*(u*ux+v*uy)+pu-TRUTH[1]*lapu
        rv = vt+TRUTH[0]*(u*vx+v*vy)+pv-TRUTH[1]*lapv
        omega = np.fft.ifft2(wh).real
        velocity.append(np.stack([u, v], -1))
        pressure.append(p)
        vorticity.append(omega)
        energy.append(float(np.mean(u*u+v*v)/2))
        enstrophy.append(float(np.mean(omega*omega)/2))
        advection.append(np.fft.ifft2(advhat).real.ravel())
        diffusion.append(np.fft.ifft2(-k2*wh).real.ravel())
        time_derivative.append(np.fft.ifft2(wt).real.ravel())
        divergence.append(float(np.max(np.abs(ux+vy))))
        momentum_rms.append(float(np.sqrt(np.mean(ru*ru+rv*rv))))
    design = np.column_stack([-np.concatenate(advection), np.concatenate(diffusion)])
    fit = np.linalg.lstsq(design, np.concatenate(time_derivative), rcond=None)[0]
    audit = {'grid': n, 'rtol': rtol, 'atol': atol, 'integrator': 'scipy.solve_ivp DOP853',
             'nfev': solution.nfev, 'elapsed_seconds': time.perf_counter()-start,
             'initial_streamfunction': '0.5 sin(x)sin(y)+0.15 cos(2x)sin(y)+0.1 sin(x)cos(3y)',
             'uniform_velocity': [mean_u, mean_v], 'dealiasing': 'strict 2/3 spectral truncation',
             'divergence_max': max(divergence), 'momentum_residual_rms_max': max(momentum_rms),
             'energy_first_last': [energy[0], energy[-1]], 'enstrophy_first_last': [enstrophy[0], enstrophy[-1]],
             'energy_nonincreasing': bool(np.max(np.diff(energy)) < 1e-10),
             'enstrophy_nonincreasing': bool(np.max(np.diff(enstrophy)) < 1e-10),
             'coefficient_design_rank': int(np.linalg.matrix_rank(design)),
             'coefficient_design_condition': float(np.linalg.cond(design)),
             'coefficient_fit_using_solver_rhs_not_independent_evidence': fit.tolist()}
    return np.array(velocity), np.array(pressure), np.array(vorticity), audit


def prepare_numerical():
    times = np.linspace(0, 1., 41)
    coarse = spectral_solution(32, times, 2e-8, 2e-10)
    fine = spectral_solution(64, times, 2e-10, 2e-12)
    common = fine[0][:, ::2, ::2]
    relative = float(np.linalg.norm(coarse[0]-common)/np.linalg.norm(common))
    audit = {'coarse': coarse[3], 'fine': fine[3], 'common_grid_velocity_relative_difference': relative,
             'independent_of_neural_models': True, 'truth_used_only_to_generate_reference': True}
    assert relative < 1e-4 and fine[3]['momentum_residual_rms_max'] < 1e-7, audit
    assert fine[3]['coefficient_design_rank'] == 2 and fine[3]['coefficient_design_condition'] < 100
    assert fine[3]['energy_nonincreasing'] and fine[3]['enstrophy_nonincreasing']
    save_json(ROOT / 'independent_solution_grid_and_equation_checks.json', audit)
    axis = np.arange(64)*2*np.pi/64
    x, y = np.meshgrid(axis, axis, indexing='ij')
    coords = np.column_stack([np.repeat(x.ravel(), len(times)), np.repeat(y.ravel(), len(times)), np.tile(times, x.size)])
    velocity = fine[0].transpose(1, 2, 0, 3).reshape(-1, 2)
    pressure = fine[1].transpose(1, 2, 0).reshape(-1, 1)
    omega = fine[2].transpose(1, 2, 0).reshape(-1, 1)
    np.savez_compressed(ROOT / 'independent_solution_diagnostic_data.npz', coordinates=coords, velocity=velocity, pressure=pressure,
                        vorticity=omega, truth=TRUTH, bounds=np.array([[0., 0., 0.], [2*np.pi, 2*np.pi, 1.]]))
    print(json.dumps(audit, indent=2), flush=True)


class ChebyshevLayer(nn.Module):
    def __init__(self, inp, out, degree=8):
        super().__init__()
        self.degree = degree
        self.coeff = nn.Parameter(torch.randn(inp, out, degree)*.02)

    def forward(self, x):
        basis = [torch.ones_like(x), x]
        for _ in range(2, self.degree):
            basis.append(2*x*basis[-1]-basis[-2])
        return torch.einsum('nik,iok->no', torch.stack(basis, -1), self.coeff)


class DiagnosticNS(nn.Module):
    def __init__(self, bounds, hybrid):
        super().__init__()
        self.register_buffer('lower', torch.tensor(bounds[0], dtype=torch.float32))
        self.register_buffer('upper', torch.tensor(bounds[1], dtype=torch.float32))
        self.mlp = nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.hybrid = hybrid
        if hybrid:
            self.kan = nn.Sequential(ChebyshevLayer(3, 24), nn.Tanh(), ChebyshevLayer(24, 24), nn.Tanh(), nn.Linear(24, 2))
            nn.init.zeros_(self.kan[-1].weight)
            nn.init.zeros_(self.kan[-1].bias)
        self.log_lambda = nn.Parameter(torch.tensor([math.log(1.5), math.log(.02)], dtype=torch.float32))

    def forward(self, physical):
        normalized = 2*(physical-self.lower)/(self.upper-self.lower)-1
        result = self.mlp(normalized)
        if self.hybrid:
            result = result + self.kan(normalized)
        return result

    @property
    def coefficients(self):
        return self.log_lambda.exp()


def derivative(value, coordinates):
    return torch.autograd.grad(value, coordinates, torch.ones_like(value), create_graph=True)[0]


def velocity_fields(model, coordinates):
    output = model(coordinates)
    grad_psi = derivative(output[:, :1], coordinates)
    return grad_psi[:, 1:2], -grad_psi[:, :1], output[:, 1:2]


def momentum_residual(model, coordinates):
    u, v, p = velocity_fields(model, coordinates)
    du, dv, dp = derivative(u, coordinates), derivative(v, coordinates), derivative(p, coordinates)
    lap_u = derivative(du[:, :1], coordinates)[:, :1] + derivative(du[:, 1:2], coordinates)[:, 1:2]
    lap_v = derivative(dv[:, :1], coordinates)[:, :1] + derivative(dv[:, 1:2], coordinates)[:, 1:2]
    l1, l2 = model.coefficients
    fu = du[:, 2:3] + l1*(u*du[:, :1]+v*du[:, 1:2]) + dp[:, :1] - l2*lap_u
    fv = dv[:, 2:3] + l1*(u*dv[:, :1]+v*dv[:, 1:2]) + dp[:, 1:2] - l2*lap_v
    return fu, fv, du[:, :1]+dv[:, 1:2]


def prepare_split(case, n_train=10000, n_test=20000):
    path = ROOT / f'{case}_diagnostic_data.npz'
    data = np.load(path)
    split_path = ROOT / f'{case}_fixed_split.npz'
    rng = np.random.default_rng(SEED)
    indexes = rng.permutation(len(data['coordinates']))
    train, test = indexes[:n_train], indexes[n_train:n_train+n_test]
    if split_path.exists():
        old = np.load(split_path)
        assert np.array_equal(old['train'], train) and np.array_equal(old['test'], test)
    else:
        np.savez_compressed(split_path, train=train, test=test)
    assert len(np.intersect1d(train, test)) == 0
    return data, train, test


def evaluate(model, data, test, device):
    predictions, pressures, divergences, pde = [], [], [], []
    for start in range(0, len(test), 2048):
        coordinates = torch.tensor(data['coordinates'][test[start:start+2048]], dtype=torch.float32, device=device, requires_grad=True)
        u, v, p = velocity_fields(model, coordinates)
        predictions.append(torch.cat([u, v], 1).detach().cpu().numpy())
        pressures.append(p.detach().cpu().numpy())
    pred, pp = np.concatenate(predictions), np.concatenate(pressures)
    target = data['velocity'][test]
    for start in range(0, min(2048, len(test)), 256):
        z = torch.tensor(data['coordinates'][test[start:start+256]], dtype=torch.float32, device=device, requires_grad=True)
        fu, fv, div = momentum_residual(model, z)
        divergences.append(div.detach().cpu().numpy())
        pde.append(torch.cat([fu, fv], 1).detach().cpu().numpy())
    # Pressure is identifiable only up to a time-dependent spatial constant.
    aligned = pp.copy()
    times = data['coordinates'][test, 2]
    for tt in np.unique(times):
        mask = times == tt
        aligned[mask] += np.mean(data['pressure'][test][mask]-pp[mask])
    coeff = model.coefficients.detach().cpu().numpy()
    metrics = {'velocity_relative_L2_percent': float(np.linalg.norm(pred-target)/np.linalg.norm(target)*100),
               'u_relative_L2_percent': float(np.linalg.norm(pred[:, 0]-target[:, 0])/np.linalg.norm(target[:, 0])*100),
               'v_relative_L2_percent': float(np.linalg.norm(pred[:, 1]-target[:, 1])/np.linalg.norm(target[:, 1])*100),
               'pressure_gauge_aligned_relative_L2_percent': float(np.linalg.norm(aligned-data['pressure'][test])/np.linalg.norm(data['pressure'][test])*100),
               'lambda': coeff.tolist(), 'lambda_relative_error_percent': (np.abs(coeff/TRUTH-1)*100).tolist(),
               'held_out_momentum_rms': float(np.sqrt(np.mean(np.concatenate(pde)**2))),
               'held_out_divergence_max': float(np.max(np.abs(np.concatenate(divergences)))),
               'test_points': len(test), 'pde_test_points': min(2048, len(test))}
    return metrics, pred, pp, aligned


def train(case, hybrid, pretrain, joint, lbfgs):
    assert torch.cuda.is_available(), 'gpupytorch CUDA is required'
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    name = 'parallelChebyshev_KAN' if hybrid else 'MLP'
    out = ROOT / f'{case}_{name}_random_seed42'
    out.mkdir(exist_ok=True)
    assert not (out / 'experiment_results.json').exists(), 'Completed experiment exists; preserve it, choose a separate run directory'
    data, train_idx, test_idx = prepare_split(case)
    z_data = torch.tensor(data['coordinates'][train_idx], dtype=torch.float32, device=device)
    uv_data = torch.tensor(data['velocity'][train_idx], dtype=torch.float32, device=device)
    model = DiagnosticNS(data['bounds'], hybrid).to(device)
    params = [p for key, p in model.named_parameters() if key != 'log_lambda']
    batch_size, n_f = 1024, 384
    rng = torch.Generator(device=device).manual_seed(SEED+19)
    # Collocation points are actual downstream-domain points, not outside fluid.
    candidate = torch.tensor(data['coordinates'], dtype=torch.float32, device=device)
    scale_uv = uv_data.square().mean().detach()
    scale_pde = torch.tensor(1., device=device)
    config = {'case': case, 'model': name, 'architecture': 'parallel additive MLP + Chebyshev KAN' if hybrid else 'MLP only',
              'outputs': ['psi', 'pressure'], 'observations': ['u', 'v'], 'seed': SEED,
              'pretrain_steps': pretrain, 'joint_steps': joint, 'lbfgs_max_iterations': lbfgs,
              'label_points': len(train_idx), 'batch_size': batch_size, 'collocation_batch': n_f,
              'lambda_init': [1.5, .02], 'lambda_truth_used_in_loss': False, 'lambda_prior_weight': 0,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'input_normalization_inside_forward': True, 'autograd_with_respect_to': 'original physical x,y,t',
              'data_sha256': sha(ROOT / f'{case}_diagnostic_data.npz'), 'split_sha256': sha(ROOT / f'{case}_fixed_split.npz'),
              'device': torch.cuda.get_device_name(device), 'torch_version': torch.__version__,
              'loss': 'velocity MSE / training velocity mean square + momentum MSE; no parameter truth/pressure labels/BC priors',
              'scope': 'single-seed matched-iteration diagnostic; not parameter-matched or full publication validation'}
    save_json(out / 'training_config.json', config)
    history = []
    start = time.perf_counter()
    print('RUN', case, name, json.dumps(config, ensure_ascii=False), flush=True)

    def data_loss(indices):
        z = z_data[indices].detach().requires_grad_(True)
        u, v, _ = velocity_fields(model, z)
        return ((torch.cat([u, v], 1)-uv_data[indices])**2).mean()/scale_uv

    def pde_loss(z):
        fu, fv, _ = momentum_residual(model, z.detach().requires_grad_(True))
        return (fu.square().mean()+fv.square().mean())/2/scale_pde

    def record(stage, step, loss, dl, pl):
        row = {'stage': stage, 'step': step, 'loss': float(loss.detach()), 'data_loss': float(dl.detach()),
               'pde_loss': float(pl.detach()) if torch.is_tensor(pl) else float(pl),
               'lambda1': float(model.coefficients[0].detach()), 'lambda2': float(model.coefficients[1].detach()),
               'elapsed_seconds': time.perf_counter()-start}
        history.append(row)
        print(json.dumps(row), flush=True)

    model.log_lambda.requires_grad_(False)
    opt = torch.optim.Adam(params, lr=1e-3)
    for step in range(1, pretrain+1):
        idx = torch.randint(len(z_data), (batch_size,), device=device, generator=rng)
        opt.zero_grad(set_to_none=True)
        dl = data_loss(idx)
        assert torch.isfinite(dl)
        dl.backward()
        opt.step()
        if step == 1 or step % 500 == 0 or step == pretrain:
            record('supervised', step, dl, dl, 0.)
    model.log_lambda.requires_grad_(True)
    opt = torch.optim.Adam([{'params': params, 'lr': 5e-4}, {'params': [model.log_lambda], 'lr': 2e-3}])
    for step in range(1, joint+1):
        idx = torch.randint(len(z_data), (batch_size,), device=device, generator=rng)
        ids = torch.randint(len(candidate), (n_f,), device=device, generator=rng)
        opt.zero_grad(set_to_none=True)
        dl, pl = data_loss(idx), pde_loss(candidate[ids])
        loss = dl + pl
        assert torch.isfinite(loss), (case, name, step)
        loss.backward()
        opt.step()
        if step == 1 or step % 500 == 0 or step == joint:
            record('joint', step, loss, dl, pl)
    if lbfgs:
        fixed_data = torch.randperm(len(z_data), device=device, generator=rng)[:2048]
        fixed_physics = candidate[torch.randperm(len(candidate), device=device, generator=rng)[:768]]
        opt = torch.optim.LBFGS(model.parameters(), lr=.8, max_iter=lbfgs, max_eval=lbfgs*2,
                               history_size=50, line_search_fn='strong_wolfe', tolerance_grad=1e-9, tolerance_change=1e-11)
        calls = [0]

        def closure():
            opt.zero_grad(set_to_none=True)
            dl, pl = data_loss(fixed_data), pde_loss(fixed_physics)
            loss = dl + pl
            assert torch.isfinite(loss)
            loss.backward()
            calls[0] += 1
            if calls[0] == 1 or calls[0] % 100 == 0:
                record('lbfgs', calls[0], loss, dl, pl)
            return loss

        opt.step(closure)
        config['lbfgs_actual_closure_calls'] = calls[0]
    torch.cuda.synchronize()
    metrics, pred, p, aligned = evaluate(model, data, test_idx, device)
    metrics.update({'case': case, 'model': name, 'elapsed_seconds': time.perf_counter()-start,
                    'parameter_count': config['parameter_count'], 'selected_checkpoint': 'final; no test-based checkpoint selection',
                    'diagnostic_only': True, 'original_NS_results_reusable': False})
    torch.save(model.state_dict(), out / 'final_weights.pt')
    np.savez_compressed(out / 'fixed_test_predictions.npz', indices=test_idx, coordinates=data['coordinates'][test_idx],
                        reference_velocity=data['velocity'][test_idx], predicted_velocity=pred,
                        reference_pressure=data['pressure'][test_idx], predicted_pressure=p, aligned_pressure=aligned)
    with (out / 'training_record.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    save_json(out / 'training_config.json', config)
    save_json(out / 'experiment_results.json', metrics)
    print('RESULT', json.dumps(metrics, ensure_ascii=False), flush=True)


def self_test():
    class Analytic(nn.Module):
        coefficients = torch.tensor([1., .01], dtype=torch.float64, device='cuda')

        def forward(self, z):
            x, y, t = z[:, :1], z[:, 1:2], z[:, 2:3]
            a = torch.exp(-.02*t)
            psi = a*torch.sin(x)*torch.sin(y)
            p = a*a*(torch.cos(2*x)+torch.cos(2*y))/4
            return torch.cat([psi, p], 1)

    torch.manual_seed(SEED)
    z = torch.rand(64, 3, dtype=torch.float64, device='cuda', requires_grad=True)*2
    fu, fv, div = momentum_residual(Analytic(), z)
    assert max(float(fu.abs().max()), float(fv.abs().max()), float(div.abs().max())) < 1e-12
    values = {'analytic_Taylor_Green_momentum_max': float(torch.max(torch.abs(torch.cat([fu, fv], 1)))),
              'analytic_divergence_max': float(div.abs().max())}
    for hybrid in (False, True):
        model = DiagnosticNS(np.array([[1., -2., 0.], [8., 2., 5.]]), hybrid).cuda().double()
        coordinates = torch.rand(16, 3, dtype=torch.float64, device='cuda', requires_grad=True)
        fu, fv, div = momentum_residual(model, coordinates)
        loss = fu.square().mean()+fv.square().mean()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(model.log_lambda.grad).all()
        assert float(div.abs().max()) < 1e-12
        # The normalized-coordinate analytic function checks every chain factor.
        lower, upper = model.lower, model.upper
        h = 2/(upper-lower)
        q = 2*(coordinates-lower)/(upper-lower)-1
        value = q[:, :1]**3 + q[:, 1:2]**3 + q[:, 2:3]**3
        d1 = derivative(value, coordinates)
        assert torch.allclose(d1, 3*q*q*h, atol=1e-12, rtol=1e-12)
        for axis in range(3):
            d2 = derivative(d1[:, axis:axis+1], coordinates)[:, axis:axis+1]
            d3 = derivative(d2, coordinates)[:, axis:axis+1]
            assert torch.allclose(d2, 6*q[:, axis:axis+1]*h[axis]**2, atol=1e-12, rtol=1e-12)
            assert torch.allclose(d3, torch.full_like(d3, float(6*h[axis]**3)), atol=1e-12, rtol=1e-12)
        values['hybrid' if hybrid else 'mlp'] = {'finite_backward': True, 'divergence_max': float(div.abs().max()),
                                                  'physical_coordinate_chain_orders_1_2_3_passed': True}
    save_json(ROOT / 'CUDAphysics_derivative_and_residual_checks.json', values)
    print(json.dumps(values, indent=2), flush=True)


def run_all(args):
    self_test()
    if not (ROOT / 'cylinder_wake_source_and_equation_checks.json').exists():
        prepare_wake()
    if not (ROOT / 'independent_solution_grid_and_equation_checks.json').exists():
        prepare_numerical()
    for case in ('cylinder_wake', 'independent_solution'):
        for hybrid in (False, True):
            name = 'parallelChebyshev_KAN' if hybrid else 'MLP'
            folder = ROOT / f'{case}_{name}_random_seed42'
            if (folder / 'experiment_results.json').exists():
                old = json.loads((folder / 'training_config.json').read_text(encoding='utf-8'))
                assert (old['pretrain_steps'], old['joint_steps'], old['lbfgs_max_iterations']) == (args.pretrain, args.joint, args.lbfgs), 'Budget differs: preserve prior run and use a separate output directory'
                assert old['data_sha256'] == sha(ROOT / f'{case}_diagnostic_data.npz'), 'Reference data changed since this run'
                assert old['split_sha256'] == sha(ROOT / f'{case}_fixed_split.npz'), 'Label split changed since this run'
                print('Preserving completed run', case, name, flush=True)
                continue
            train(case, hybrid, args.pretrain, args.joint, args.lbfgs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'self-test', 'all'])
    parser.add_argument('--pretrain', type=int, default=3000)
    parser.add_argument('--joint', type=int, default=6000)
    parser.add_argument('--lbfgs', type=int, default=400)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare_wake()
        prepare_numerical()
    elif args.mode == 'self-test':
        self_test()
    else:
        run_all(args)
