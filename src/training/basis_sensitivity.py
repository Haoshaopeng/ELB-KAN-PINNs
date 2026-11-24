"""Bounded, three-seed follow-up. Frozen science modules are never edited."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)
CORE = ('MLP', 'Adaptive', 'Uniform', 'SharedLearned')
CONDITIONS = ('noise_1pct', 'observations_25pct')


def jobs():
    result = []
    for condition in CONDITIONS:
        for case in ('KdV', 'NS_periodic'):
            for variant in CORE:
                for seed in SEEDS:
                    result.append(dict(id='%s_%s_%s_seed%s' % (condition, case, variant, seed),
                        condition=condition, mode='inverse', job=dict(case=case, variant=variant, seed=seed)))
    for seed in SEEDS:
        result.append(dict(id='fourier_4terms_KdV_Fourier_seed%s' % seed,
            condition='fourier_4terms', mode='forward', job=dict(case='KdV', variant='Fourier', seed=seed)))
    assert len(result) == 51 and len({r['id'] for r in result}) == 51
    return result


def transform_observations(z, y, indices, case, condition, seed):
    import numpy as np
    assert condition in CONDITIONS and seed in SEEDS
    generator = np.random.default_rng(20261017 + seed + (10000 if case == 'NS_periodic' else 0))
    selected = np.arange(len(z))
    if condition == 'observations_25pct':
        selected = np.sort(generator.permutation(len(z))[:len(z)//4])
    clean = np.asarray(y[selected], dtype=np.float32)
    observed = clean.copy()
    if condition == 'noise_1pct':
        # Known hard initial conditions remain exact; only interior observations are noisy.
        interior = z[selected, -1] > 0
        rms = np.sqrt(np.mean(clean[interior].astype(np.float64)**2, axis=0, keepdims=True))
        observed[interior] = (clean[interior] + .01*rms*generator.standard_normal(clean[interior].shape)).astype(np.float32)
    return np.asarray(z[selected], dtype=np.float32), observed, np.asarray(indices[selected])


def install(spec):
    import numpy as np
    import torch
    import adaptive_basis as study
    import forward_training as full
    import inverse_training as inverse
    import shared_gate_control as shared
    assert spec in jobs(), 'Unknown or modified follow-up task'
    condition, case, variant, seed = (spec['condition'], spec['job']['case'], spec['job']['variant'], spec['job']['seed'])
    if variant == 'SharedLearned':
        shared.install()
    elif spec['mode'] == 'forward':
        full.protocol_test()
        study.config, study.digest = full.config, full.digest
    module = inverse if spec['mode'] == 'inverse' else study
    old_config, old_digest = module.config, module.digest
    original_hashes = old_digest()
    extension_hash = study.base.sha(Path(__file__))
    cached = {}

    def config(name, smoke=False):
        cfg = dict(old_config(name, smoke))
        cfg['followup_condition'] = condition
        if spec['mode'] == 'inverse':
            cfg.update(noise=.01 if condition == 'noise_1pct' else 0.,
                observation_fraction=.25 if condition == 'observations_25pct' else 1.,
                observation_seed=20261017+seed+(10000 if name == 'NS_periodic' else 0),
                observation_noise_definition='interior observations: independent Gaussian, 1% training-interior channel RMS; t=0 unchanged',
                test_split='unchanged original holdout; omitted training points NOT added to holdout')
        else:
            cfg.update(input_fourier_terms=4, input_max_harmonic=2,
                comparison='Same per-family harmonics as mixed bank; NOT equal total parameter count')
        return cfg

    def digest():
        assert old_digest() == original_hashes
        assert study.base.sha(Path(__file__)) == extension_hash
        return dict(original_hashes, **{Path(__file__).name: extension_hash})

    if spec['mode'] == 'inverse':
        old_hashes = inverse.data_hashes

        def arrays(name):
            assert name == case
            if name not in cached:
                with np.load(inverse.DATA/(name+'_observations.npz')) as data, np.load(inverse.DATA/(name+'_split.npz')) as split:
                    assert np.array_equal(data['indices'], split['train'])
                    assert not np.intersect1d(data['indices'], split['test']).size
                    changed = transform_observations(data['coordinates'], data['target'], data['indices'], name, condition, seed)
                    assert not np.intersect1d(changed[2], split['test']).size
                    cached[name] = changed
            return cached[name]

        def data_hashes(name):
            hashes = dict(old_hashes(name))
            for key, array in zip(('coordinates', 'observed_target', 'selected_original_indices'), arrays(name)):
                hashes['followup_'+key] = hashlib.sha256(array.tobytes(order='C')).hexdigest()
            return hashes

        def load_observations(name, device):
            z, y, _ = arrays(name)
            z = torch.tensor(z, device=device, dtype=torch.float32)
            y = torch.tensor(y, device=device, dtype=torch.float32)
            scale = y.square().mean().detach()
            assert torch.isfinite(scale) and scale > 0
            return z, y, scale

        inverse.load_observations, inverse.data_hashes = load_observations, data_hashes
    else:
        old_model = study.AdaptiveModel

        class NarrowFourier(old_model):
            def __init__(self, name, kind):
                assert name == 'KdV' and kind == 'Fourier'
                super().__init__(name, kind)
                old = self.kan[0]
                state = torch.get_rng_state()
                new = study.VariableBasisLayer(2, 32, 4, 'Fourier')
                torch.set_rng_state(state)
                # Retain the original coefficients for k=1,2, including their sine terms.
                with torch.no_grad():
                    new.coeff.copy_(old.coeff[:, :, [0, 1, 8, 9]])
                self.kan[0] = new

        study.AdaptiveModel = NarrowFourier
    module.config, module.digest = config, digest
    return module


def main():
    import platform
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    assert platform.system() == 'Linux', 'Cloud training only'
    assert torch.cuda.is_available() and torch.__version__ == '2.5.1+cu121'
    spec = next(j for j in jobs() if j['id'] == args.task)
    module = install(spec)
    import adaptive_basis as study
    smoke = args.root/'smoke'
    smoke.mkdir(parents=True, exist_ok=True)
    module.train(spec['job'], smoke, True)
    # Each formal task starts again from its original seed, not from the smoke checkpoint.
    module.train(spec['job'], args.root, False)
    study.base.save(args.root/'complete.json', dict(task=spec, smoke_passed=True, formal_complete=True))


if __name__ == '__main__':
    main()
