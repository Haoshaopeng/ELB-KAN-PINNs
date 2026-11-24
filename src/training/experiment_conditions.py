"""Isolated adapters for the 135 non-wake paper conditions. No frozen edits."""
import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

import adaptive_basis as study
import inverse_training as inverse
import burgers_training as burgers
from architecture_controls import hidden_layer

HERE = Path(__file__).resolve().parent
MODELS = ('MLP', 'Chebyshev', 'Fourier', 'RBF', 'Wavelet', 'Uniform',
          'Adaptive', 'SharedLearned', 'Fourier_HFourier')
BASE_MODEL = study.AdaptiveModel
INV_CONFIG = inverse.config
INV_DIGEST = inverse.digest
INV_DATA_HASHES = inverse.data_hashes
INV_EVALUATE = inverse.evaluate
BASE_CHECKPOINT = study.checkpoint
BASE_LBFGS = torch.optim.LBFGS
INSTALLED = None


def jobs():
    groups = [('burgers_focus', 'Burgers', 'focus_50pct', 'forward'),
              ('kdv_noise5', 'KdV', 'noise_5pct', 'inverse'),
              ('periodic_noise5', 'NS_periodic', 'noise_5pct', 'inverse'),
              ('kdv_init', 'KdV', 'alternative_initialization', 'inverse'),
              ('periodic_init', 'NS_periodic', 'alternative_initialization', 'inverse')]
    return [dict(id=f'paper_{group}_{v}_seed{s}', kind='paper', group=group,
                 condition=condition, mode=mode, job=dict(case=case, variant=v, seed=s))
            for group, case, condition, mode in groups for v in MODELS for s in (42, 43, 44)]


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def observations(spec):
    case, seed = spec['job']['case'], spec['job']['seed']
    with np.load(inverse.DATA / (case + '_observations.npz')) as data:
        z, clean, indices = data['coordinates'].copy(), data['target'].copy(), data['indices'].copy()
    with np.load(inverse.DATA / (case + '_split.npz')) as split:
        assert np.array_equal(indices, split['train'])
        assert not np.intersect1d(indices, split['test']).size
    y = clean.copy()
    mask = z[:, -1] > 0
    scales = clean[mask].astype(np.float64).std(axis=0, ddof=0)
    if spec['condition'] == 'noise_5pct':
        assert np.all(scales > 0)
        rng = np.random.default_rng(20261018 + seed + (10000 if case == 'NS_periodic' else 0))
        y[mask] += (.05 * scales * rng.standard_normal(y[mask].shape)).astype(y.dtype)
    assert np.array_equal(y[~mask], clean[~mask])
    meta = dict(condition=spec['condition'], coordinates_sha256=array_sha(z),
                indices_sha256=array_sha(indices), labels_sha256=array_sha(y),
                clean_labels_sha256=array_sha(clean), noise_component_std=scales.tolist(),
                noise_relative_std=.05 if spec['condition'] == 'noise_5pct' else 0.,
                observation_points=len(z), unchanged_initial_observations=True,
                observation_holdout_disjoint=True)
    return z, y, meta


class Network(BASE_MODEL):
    def __init__(self, case, variant):
        source = 'Adaptive' if variant == 'SharedLearned' else 'Fourier' if variant == 'Fourier_HFourier' else variant
        if case == 'Burgers':
            # Construction is identical to the current Burgers baseline.
            nn.Module.__init__(self)
            template = burgers.BurgersModel(case, 'Fourier' if variant == 'Fourier_HFourier' else variant)
            self.__dict__.update(template.__dict__)
        else:
            super().__init__(case, source)
            if variant == 'SharedLearned':
                self.kan[0].logits = nn.Parameter(self.kan[0].logits[:1].detach().clone())
        if variant == 'Fourier_HFourier':
            self.kan[2] = hidden_layer(self.kan[2], 'Fourier')
        self.case, self.variant = case, variant

    def correction(self, z):
        if self.case == 'Burgers':
            return burgers.BurgersModel.correction(self, z)
        return super().correction(z)

    def forward(self, z):
        if self.case == 'Burgers':
            return burgers.BurgersModel.forward(self, z)
        return super().forward(z)


def focus_sample(case, count, bc_count, generator):
    z, bc = burgers.sample(case, count, bc_count, generator)
    z = z.detach()
    # Same random draws and total count as the uniform paired control.
    z[:count // 2, 0] *= .4
    return z.requires_grad_(), bc


def install(spec):
    global INSTALLED
    assert INSTALLED is None, 'One scientific adapter per worker process'
    assert spec in jobs()
    INSTALLED = spec
    case = spec['job']['case']
    original_hashes = INV_DIGEST() if case != 'Burgers' else burgers.digest()
    original_hashes[Path(__file__).name] = study.base.sha(Path(__file__))
    original_hashes['architecture_controls.py'] = study.base.sha(HERE / 'architecture_controls.py')
    if case == 'Burgers':
        burgers.install()
        study.AdaptiveModel = Network
        study.base.sample = focus_sample
        def cfg(c, smoke=False):
            value = burgers.config(c, smoke)
            return dict(value, sampling='50% uniform x[-.4,.4], 50% uniform x[-1,1]; t uniform',
                        importance_reweighting=False, paper_condition=spec['condition'])
        study.config = cfg
        study.digest = lambda: dict(original_hashes)
        module = study
    else:
        study.AdaptiveModel = Network
        z, y, meta = observations(spec)
        if spec['condition'] == 'alternative_initialization':
            inverse.INITIAL[case] = [.001] if case == 'KdV' else [.5, .005]
        def cfg(c, smoke=False):
            value = INV_CONFIG(c, smoke)
            return dict(value, paper_condition=spec['condition'],
                        noise=.05 if spec['condition'] == 'noise_5pct' else 0.,
                        observation_transform=meta)
        def loader(c, device):
            assert c == case
            zz = torch.tensor(z, device=device, dtype=torch.float32)
            yy = torch.tensor(y, device=device, dtype=torch.float32)
            return zz, yy, yy.square().mean().detach()
        inverse.config = cfg
        inverse.load_observations = loader
        inverse.data_hashes = lambda c: dict(INV_DATA_HASHES(c), transform=meta)
        inverse.digest = lambda: dict(original_hashes)
        def evaluate(model, folder, smoke):
            result = INV_EVALUATE(model, folder, smoke)
            if model.variant == 'SharedLearned':
                result['gate_diagnostic'] = dict(variables=['shared'], families=study.FAMILIES,
                    weights=model.kan[0].weights().detach().cpu().tolist(),
                    warning='Free coefficients absorb gates; not causal physical importance')
            return result
        inverse.evaluate = evaluate
        module = inverse
    return module


def self_test(spec, device='cpu'):
    torch.set_num_threads(1)
    case, variant, seed = (spec['job'][k] for k in ('case', 'variant', 'seed'))
    study.base.seed(seed)
    factory = study.AdaptiveModel if case == 'Burgers' else inverse.InverseModel
    model = factory(case, variant).to(device).double()
    with torch.no_grad():
        for p in model.parameters():
            if p is not getattr(model, 'log_coefficients', None):
                p.add_(.003 * torch.randn_like(p))
    dim = 2 if case in ('Burgers', 'KdV') else 3
    z = torch.rand(5, dim, dtype=torch.float64, device=device).requires_grad_()
    scale = torch.linspace(.4, 1.1, dim, dtype=torch.float64, device=device)
    q = ((z.detach() - .17) / scale).requires_grad_()
    errors = []
    for channel in range(model(z).shape[1]):
        for axis in range(dim):
            a, b = model(z)[:, channel:channel+1], model(q * scale + .17)[:, channel:channel+1]
            for _ in range(3):
                a = study.base.derivative(a, z)[:, axis:axis+1]
                b = study.base.derivative(b, q)[:, axis:axis+1] / scale[axis]
                assert torch.allclose(a, b, rtol=1e-7, atol=1e-8)
                errors.append(float((a-b).abs().max().detach()))
    objective = (burgers.residual(model, z) if case == 'Burgers' else inverse.residual(model, z)).square().mean()
    objective.backward()
    assert torch.isfinite(objective)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    coefficients = None
    if case != 'Burgers':
        coefficients = model.log_coefficients.exp().detach().cpu().tolist()
        assert np.allclose(coefficients, inverse.config(case)['initial_unknown_coefficients'])
        assert torch.all(model.log_coefficients.grad.abs() > 1e-12)
        zobs, yobs, meta = observations(spec)
        assert meta['unchanged_initial_observations'] and meta['observation_holdout_disjoint']
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    restored = factory(case, variant).to(device).double()
    restored.load_state_dict(state)
    assert torch.equal(model(z), restored(z))
    return dict(passed=True, spec=spec, optimizer_steps=0, device=device,
                physical_derivative_max_difference=max(errors), finite_backward=True,
                reload_exact=True, initial_coefficients=coefficients,
                source_sha256=study.base.sha(Path(__file__)))


def train(spec, root, smoke=False):
    assert platform.system() == 'Linux' and torch.cuda.is_available(), 'Cloud training only'
    assert INSTALLED == spec
    job = spec['job']; folder = root / f"{job['case']}_{job['variant']}_seed{job['seed']}"
    folder.mkdir(parents=True, exist_ok=True)
    module = study if spec['mode'] == 'forward' else inverse
    snapshots = folder / 'snapshots'; snapshots.mkdir(exist_ok=True)
    saved_model = []
    factory_name = 'AdaptiveModel' if spec['mode'] == 'forward' else 'InverseModel'
    original_factory = getattr(module, factory_name)
    def rng_state():
        return dict(torch_cpu_rng=torch.get_rng_state(), torch_cuda_rng=torch.cuda.get_rng_state_all(),
                    numpy_rng=np.random.get_state(), python_rng=random.getstate())
    def factory(*args, **kwargs):
        model = original_factory(*args, **kwargs)
        saved_model.append(model)
        path = snapshots / 'initial.pt'
        assert not path.exists()
        BASE_CHECKPOINT(path, dict(model=model.state_dict(), stage='initial', step=0,
            spec=spec, config=module.config(job['case'], smoke), optimizer=None, **rng_state()))
        return model
    def archive(path, value):
        BASE_CHECKPOINT(path, value)
        if path.name == 'checkpoint.pt':
            target = snapshots / (value['stage'] + '_' + str(value['step']) + '.pt')
            assert not target.exists(), 'Snapshot overwrite prohibited'
            BASE_CHECKPOINT(target, dict(value, **rng_state()))
    class ArchivedLBFGS(BASE_LBFGS):
        def step(self, closure):
            result = super().step(closure)
            BASE_CHECKPOINT(snapshots / 'LBFGS_accepted_final.pt', dict(
                model=saved_model[-1].state_dict(), optimizer=self.state_dict(),
                stage='L-BFGS accepted final', spec=spec, **rng_state()))
            return result
    setattr(module, factory_name, factory)
    study.checkpoint = archive
    torch.optim.LBFGS = ArchivedLBFGS
    result = module.train(job, root, smoke)
    manifest = {str(p.relative_to(folder)): study.base.sha(p) for p in snapshots.glob('*.pt')}
    study.base.save(folder / 'snapshot_manifest.json', dict(files=manifest, spec=spec,
        policy='initial; each 500 Adam/pretrain steps; stage ends; accepted final LBFGS',
        closure_trials_are_not_accepted_iterations=True))
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    task = next(s for s in jobs() if s['id'] == args.task)
    install(task)
    study.base.save(args.output, self_test(task, args.device))
