"""Exploratory variable-wise smooth basis mixtures for label-free forward PDEs.

Only the first KAN layer varies. All hybrid hidden KAN layers are Chebyshev.
Soft gates are mixture weights, not identifiable selections or causal importance.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('audited_forward', HERE / 'forward_physics_loss.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
FAMILIES = ('Chebyshev', 'Fourier', 'RBF', 'Wavelet')
VARIANTS = ('MLP', 'Adaptive', 'Uniform', 'Chebyshev', 'Fourier', 'RBF', 'Wavelet')
ROOT = HERE / 'adaptive_input_basis' / 'pilot_experiments'


def basis_values(q, family, terms):
    if family == 'Chebyshev':
        values = [torch.ones_like(q), q]
        for _ in range(2, terms):
            values.append(2*q*values[-1]-values[-2])
        return torch.stack(values[:terms], -1)
    if family == 'Fourier':
        assert terms % 2 == 0
        frequencies = torch.arange(1, terms//2+1, device=q.device, dtype=q.dtype)
        # Spatial phase is physical k*pi*x (KdV) or k*x/k*y (NS), up to a phase shift.
        # Time spans a half period so the feature map does not equate t=0 and t=1.
        phase = torch.cat([math.pi*q[:, :-1], math.pi*(q[:, -1:]+1)/2], 1)
        phase = phase.unsqueeze(-1)*frequencies
        return torch.cat([phase.cos(), phase.sin()], -1)
    centers = torch.linspace(-1, 1, terms, device=q.device, dtype=q.dtype)
    width = 2 / math.sqrt(terms)
    r = (q.unsqueeze(-1)-centers) / width
    gaussian = torch.exp(-.5*r.square())
    if family == 'RBF':
        return gaussian
    if family == 'Wavelet':
        return (1-r.square())*gaussian
    raise ValueError(family)


class VariableBasisLayer(nn.Module):
    def __init__(self, inputs, outputs, total_terms, variant):
        super().__init__()
        self.families = FAMILIES if variant in ('Uniform', 'Adaptive') else (variant,)
        self.variant = variant
        self.terms = total_terms // len(self.families)
        assert self.terms*len(self.families) == total_terms
        self.coeff = nn.Parameter(torch.randn(inputs, outputs, total_terms)*.01)
        gates = torch.zeros(inputs, len(self.families))
        if variant == 'Adaptive':
            self.logits = nn.Parameter(gates)
        else:
            self.register_buffer('logits', gates)
        # Fixed quadrature on the domain, independent of any reference solution.
        q = torch.linspace(-1, 1, 2049, dtype=torch.float64)[:, None].expand(-1, inputs)
        features = torch.cat([basis_values(q, f, self.terms) for f in self.families], -1)
        means = features.mean(0)
        deviations = (features-means).square().mean(0).sqrt()
        constant = deviations < 1e-10
        self.register_buffer('means', torch.where(constant, 0., means).float())
        self.register_buffer('scales', torch.where(constant, 1., deviations).float())

    def weights(self):
        return self.logits.softmax(dim=-1)

    def forward(self, q):
        values = torch.cat([basis_values(q, f, self.terms) for f in self.families], -1)
        values = (values-self.means)/self.scales
        # The factor F makes uniform gates exactly equal the fixed mixed feature bank.
        weights = (len(self.families)*self.weights()).repeat_interleave(self.terms, dim=-1)
        return torch.einsum('nik,iok->no', values*weights, self.coeff)


class AdaptiveModel(base.ForwardModel):
    def __init__(self, case, variant):
        super().__init__(case, 'MLP' if variant == 'MLP' else 'Chebyshev')
        self.variant = variant
        if variant == 'MLP' and case == 'NS_periodic':
            # Approximately match the larger (16-term input) hybrid's parameter count.
            self.mlp = base.make_mlp(3, (84, 83, 83), 2)
        elif variant != 'MLP':
            self.kan[0] = VariableBasisLayer(2 if case == 'KdV' else 3,
                                           32 if case == 'KdV' else 24, 16, variant)

    def correction(self, z):
        if self.case == 'KdV':
            q = torch.stack([z[:, 0], 2*z[:, 1]-1], 1)
        else:
            q = torch.stack([z[:, 0]/math.pi-1, z[:, 1]/math.pi-1, 2*z[:, 2]-1], 1)
        return self.mlp(q) + (self.kan(q) if self.variant != 'MLP' else 0.)


def digest():
    paths = ['adaptive_basis.py', 'forward_physics_loss.py',
             'benchmark_training.py', 'navier_stokes_diagnostics.py']
    return {name: base.sha(HERE/name) for name in paths}


def config(case, smoke=False):
    return dict(adam_steps=6 if smoke else 2000 if case == 'KdV' else 1500,
                nf=16 if smoke else 1024 if case == 'KdV' else 384,
                nb=8 if smoke else 128 if case == 'KdV' else 64,
                lbfgs_max_iter=3 if smoke else 100, precision='FP32',
                internal_label_count=0, fixed_coefficients=[1., .0025 if case == 'KdV' else .01],
                boundary_weight=10., checkpoint='final only', smoke=smoke)


def self_test():
    reports = []
    for case in ('KdV', 'NS_periodic'):
        for variant in VARIANTS:
            base.seed(42)
            model = AdaptiveModel(case, variant).cuda().double()
            # A nonzero readout is needed to exercise input layers and gate gradients.
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(.003*torch.randn_like(p))
            rng = torch.Generator(device='cuda').manual_seed(140)
            z, bc = base.sample(case, 6, 4, rng)
            z, bc = z.double().detach().requires_grad_(), bc.double()
            loss, _, _ = base.loss_parts(model, z, bc)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            initial = z.detach().clone()
            initial[:, -1] = 0.
            initial.requires_grad_()
            if case == 'KdV':
                ic = float((model(initial)-torch.cos(math.pi*initial[:, :1])).abs().max().detach())
            else:
                u, v, _ = base.ns.velocity_fields(model, initial)
                x, y = initial[:, :1], initial[:, 1:2]
                u0 = .3+.5*x.sin()*y.cos()+.15*(2*x).cos()*y.cos()-.3*x.sin()*(3*y).sin()
                v0 = .2-.5*x.cos()*y.sin()+.3*(2*x).sin()*y.sin()-.1*x.cos()*(3*y).cos()
                ic = float(torch.cat([u-u0, v-v0], 1).abs().max().detach())
            assert ic < 1e-12
            # Independent affine coordinates with deliberately non-unit scale factors.
            scales = torch.tensor([.7, .3] if case == 'KdV' else [.7, 1.3, .3], device='cuda', dtype=torch.float64)
            offsets = torch.full_like(scales, .17)
            q = ((z.detach()-offsets)/scales).requires_grad_()
            physical = q*scales+offsets
            pred = model(physical)
            def physical_grad(value):
                return base.derivative(value, q)/scales
            if case == 'KdV':
                du = physical_grad(pred)
                uxx = physical_grad(du[:, :1])[:, :1]
                uxxx = physical_grad(uxx)[:, :1]
                manual = du[:, 1:2]+pred*du[:, :1]+.0025*uxxx
                direct = base.kdv_residual(model, z)
            else:
                dp = physical_grad(pred[:, 1:2])
                ds = physical_grad(pred[:, :1])
                u, v = ds[:, 1:2], -ds[:, :1]
                du, dv = physical_grad(u), physical_grad(v)
                lu = physical_grad(du[:, :1])[:, :1]+physical_grad(du[:, 1:2])[:, 1:2]
                lv = physical_grad(dv[:, :1])[:, :1]+physical_grad(dv[:, 1:2])[:, 1:2]
                nu = model.coefficients[1]
                manual = torch.cat([du[:, 2:3]+u*du[:, :1]+v*du[:, 1:2]+dp[:, :1]-nu*lu,
                                    dv[:, 2:3]+u*dv[:, :1]+v*dv[:, 1:2]+dp[:, 1:2]-nu*lv], 1)
                fu, fv, _ = base.ns.momentum_residual(model, z)
                direct = torch.cat([fu, fv], 1)
            chain = float((manual-direct).abs().max().detach())
            assert chain < 1e-9, (case, variant, chain)
            gate_grad = None if variant != 'Adaptive' else float(model.kan[0].logits.grad.abs().max())
            if variant == 'Adaptive':
                assert gate_grad > 1e-14
            reports.append(dict(case=case, variant=variant, hard_initial_error=ic,
                                chain_error=chain, gate_gradient_max=gate_grad,
                                finite_backward=True, parameters=sum(p.numel() for p in model.parameters())))
    # Gate mechanism control: same initialization, exactly the same initial output.
    for inputs in (2, 3):
        base.seed(144)
        a = VariableBasisLayer(inputs, 4, 16, 'Adaptive').cuda().double()
        base.seed(144)
        b = VariableBasisLayer(inputs, 4, 16, 'Uniform').cuda().double()
        q = torch.rand(8, inputs, device='cuda', dtype=torch.float64)*2-1
        assert torch.equal(a(q), b(q))
        f = VariableBasisLayer(inputs, 4, 16, 'Fourier').cuda().double()
        left, right = q.clone(), q.clone()
        left[:, 0], right[:, 0] = -1., 1.
        assert float((f(left)-f(right)).abs().max().detach()) < 1e-12
        # Gradcheck includes mixed input partials through third order via gradgradcheck.
        q = q[:2].detach().requires_grad_()
        def first_derivative(x):
            return torch.autograd.grad(a(x).sum(), x, create_graph=True)[0]
        assert torch.autograd.gradcheck(a, (q,), atol=1e-5, rtol=1e-3)
        assert torch.autograd.gradgradcheck(first_derivative, (q,), atol=1e-5, rtol=1e-3)
    return dict(passed=True, models=reports, equal_uniform_initialization=True,
                input_derivative_finite_differences=True, optimizer_steps=0)


def checkpoint(path, data):
    temporary = path.with_suffix('.tmp')
    torch.save(data, temporary)
    os.replace(temporary, path)


def gate_diagnostic(model):
    if model.variant not in ('Adaptive', 'Uniform'):
        return None
    first = model.kan[0]
    saved = first.logits.detach().clone()
    rng = torch.Generator(device='cuda').manual_seed(713)
    z, bc = base.sample(model.case, 128, 32, rng)
    z, bc = z.double().detach().requires_grad_(), bc.double()
    scores = {}
    try:
        settings = ['learned', 'uniform'] if model.variant == 'Adaptive' else ['uniform']
        for name in settings + list(FAMILIES):
            with torch.no_grad():
                first.logits.copy_(saved)
                if name == 'uniform':
                    first.logits.zero_()
                elif name in FAMILIES:
                    first.logits[:, FAMILIES.index(name)] = -1000.
            value, physics, bc_loss = base.loss_parts(model, z, bc)
            scores[name] = dict(total=float(value.detach()), physics=float(physics.detach()),
                                boundary=float(bc_loss.detach()))
    finally:
        with torch.no_grad():
            first.logits.copy_(saved)
    return dict(variables=['x', 't'] if model.case == 'KdV' else ['x', 'y', 't'],
                families=FAMILIES, weights=first.weights().detach().cpu().tolist(),
                fixed_weight_no_retraining_pde_counterfactual=scores,
                warning='Soft mixture; gates and coefficients can compensate. Omission is not retraining evidence.')


def train(job, root, smoke=False):
    cfg = config(job['case'], smoke)
    folder = root / (job['case']+'_'+job['variant']+'_seed'+str(job['seed']))
    folder.mkdir(parents=True, exist_ok=True)
    hashes = digest()
    if (folder/'result.json').exists():
        result = json.loads((folder/'result.json').read_text(encoding='utf-8'))
        assert result['source_hashes'] == hashes and result['config'] == cfg
        assert base.sha(folder/'final.pt') == result['weights_sha256']
        return result
    if (folder/'checkpoint.pt').exists():
        raise RuntimeError('Partial run preserved. Explicit resume implementation or a separate attempt is required: '+str(folder))
    base.seed(job['seed'])
    model = AdaptiveModel(job['case'], job['variant']).cuda()
    rng = torch.Generator(device='cuda').manual_seed(job['seed']+1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg['adam_steps']//2, gamma=.25)
    history = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def progress(stage, step, total, physics, bc_loss):
        row = dict(job=job, stage=stage, step=step, adam_steps=cfg['adam_steps'],
                   loss=float(total.detach()), physics=float(physics.detach()),
                   boundary=float(bc_loss.detach()), elapsed_seconds=time.perf_counter()-start)
        if job['variant'] in ('Adaptive', 'Uniform'):
            row['gates'] = model.kan[0].weights().detach().cpu().tolist()
        history.append(row)
        print(json.dumps(row), flush=True)
        base.save(root/'progress.json', dict(status='training', **row))
        base.save(folder/'history.json', history)

    for step in range(1, cfg['adam_steps']+1):
        z, bc = base.sample(job['case'], cfg['nf'], cfg['nb'], rng)
        optimizer.zero_grad(set_to_none=True)
        total, physics, periodic = base.loss_parts(model, z, bc)
        assert torch.isfinite(total), 'Nonfinite loss'
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 100 == 0 or step == cfg['adam_steps']:
            progress('Adam', step, total, physics, periodic)
        if step % 500 == 0 or step == cfg['adam_steps']:
            checkpoint(folder/'checkpoint.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), sampling_rng=rng.get_state(), step=step,
                source_hashes=hashes, config=cfg, stage='Adam', job=job))
    z, bc = base.sample(job['case'], 2*cfg['nf'], 2*cfg['nb'], rng)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=.8, max_iter=cfg['lbfgs_max_iter'],
        max_eval=cfg['lbfgs_max_iter']*2, history_size=50, line_search_fn='strong_wolfe',
        tolerance_grad=1e-8, tolerance_change=1e-10)
    calls = 0

    def closure():
        nonlocal calls
        optimizer.zero_grad(set_to_none=True)
        total, physics, periodic = base.loss_parts(model, z, bc)
        assert torch.isfinite(total)
        total.backward()
        calls += 1
        if calls == 1 or calls % 25 == 0:
            progress('L-BFGS', calls, total, physics, periodic)
        return total

    assert float(closure().detach()) == float(closure().detach())
    optimizer.step(closure)
    checkpoint(folder/'final.pt', model.state_dict())
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    peak = torch.cuda.max_memory_allocated()
    model.load_state_dict(torch.load(folder/'final.pt', weights_only=True, map_location='cuda'))
    # The verified reference is first opened after the final weights are frozen.
    metrics = base.evaluate(model, folder, smoke)
    diagnostics = gate_diagnostic(model)
    result = dict(job=job, config=cfg, source_hashes=hashes,
        weights_sha256=base.sha(folder/'final.pt'), **metrics,
        parameters=sum(p.numel() for p in model.parameters()), training_seconds=seconds,
        peak_cuda_memory_bytes=peak, closure_calls=calls, gate_diagnostic=diagnostics,
        exploratory_not_manuscript_result=True,
        gpu=torch.cuda.get_device_name(0), python=platform.python_version(), torch=torch.__version__)
    base.save(folder/'result.json', result)
    base.save(folder/'files_sha256.json', {p.name:base.sha(p) for p in folder.iterdir()
                                         if p.is_file() and p.name != 'files_sha256.json'})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--node', choices=['A', 'B', 'C'], required=True)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    assert platform.system() == 'Linux', 'Cloud only; no local training'
    assert torch.cuda.is_available() and torch.__version__ == '2.5.1+cu121'
    root = ROOT / ('smoke_'+args.node if args.smoke else args.node)
    root.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        base.save(root/'self_test.json', self_test())
        print('SELF_TEST_PASSED', flush=True)
        return
    verification = ROOT/args.node/'self_test.json'
    assert verification.exists() and json.loads(verification.read_text())['passed'], 'Run self-test first'
    seed = {'A': 42, 'B': 43, 'C': 44}[args.node]
    jobs = [dict(case=case, variant=variant, seed=seed)
            for case in ('KdV', 'NS_periodic') for variant in VARIANTS]
    if args.smoke:
        jobs = [dict(case=case, variant='Adaptive', seed=seed) for case in ('KdV', 'NS_periodic')]
    manifest = dict(node=args.node, jobs=jobs, source_hashes=digest(),
        budgets={case:config(case, args.smoke) for case in ('KdV', 'NS_periodic')},
        purpose='Exploratory short-budget forward study; no labels or lambda estimation',
        architecture='Common MLP + KAN; only 16-term input family varies; hidden KAN remains Chebyshev',
        mixture='4 families x 4 terms; shared variable-wise gates; fixed controls have 16 terms per family',
        seeds=[42,43,44], all_seeds_reported=True,
        limitations=['Approximate parameter matching, not frequency/derivative matching',
                     'Mixture weights are not identifiable importance scores',
                     'Periodic BC is soft; mixed bases are not all intrinsically periodic',
                     'Three seeds and short budgets are not confirmatory evidence'])
    path = root/'manifest.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == manifest
    else:
        base.save(path, manifest)
    # Separate advisory lock stops accidental duplicate launch on the same node.
    import fcntl
    with (root/'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = []
        for job in jobs:
            try:
                result = train(job, root, args.smoke)
            except Exception as exc:
                base.save(root/'progress.json', dict(status='failed', job=job, error=repr(exc)))
                raise
            results.append(result)
            base.save(root/'completed.json', results)
            torch.cuda.empty_cache()
        base.save(root/'progress.json', dict(status='complete', completed=len(results), jobs=len(jobs)))


if __name__ == '__main__':
    main()
