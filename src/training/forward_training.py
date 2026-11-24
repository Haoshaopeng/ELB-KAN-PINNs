"""Full-budget replication of the variable-basis pilot.

Only the budgets, output location, and provenance differ. The model, sampling,
optimization loop, and final-only evaluation come from the hashed pilot module.
"""
import argparse
import json
from pathlib import Path
import platform

import adaptive_basis as study

ROOT = study.HERE / 'adaptive_input_basis' / 'full_budget_experiments'
FROZEN_HASHES = {
    'adaptive_basis.py': 'fee10967324f8c182eb4862a91ae8ca3ad39d7c80f906977138e062981a32249',
    'forward_physics_loss.py': 'b6bcb79906ef36e5da4aa5674ff305194767433251cb655baf3539112595474c',
    'benchmark_training.py': '53a6322d6b16e9af495c228b62e7273aa1b7e1031cebd09390387052063fcd5c',
    'navier_stokes_diagnostics.py': 'da451746c75aa0aa8d881f7dbea0148d0572da19e5d3bd143b4e8253e9a5ceef',
}
PILOT_CONFIG = study.config
PILOT_DIGEST = study.digest


def config(case, smoke=False):
    cfg = PILOT_CONFIG(case, smoke)
    if not smoke:
        cfg['adam_steps'] = 8000 if case == 'KdV' else 6000
        cfg['lbfgs_max_iter'] = 400
    return cfg


def digest():
    hashes = PILOT_DIGEST()
    assert hashes == FROZEN_HASHES, 'Frozen implementation changed; do not reuse this protocol'
    hashes[Path(__file__).name] = study.base.sha(Path(__file__))
    return hashes


def protocol_test():
    digest()
    for case in ('KdV', 'NS_periodic'):
        old, full = PILOT_CONFIG(case), config(case)
        changed = {key for key in old if old[key] != full[key]}
        assert changed == {'adam_steps', 'lbfgs_max_iter'}
        assert full['adam_steps'] == 4*old['adam_steps']
        assert full['lbfgs_max_iter'] == 4*old['lbfgs_max_iter']
        assert config(case, True) == PILOT_CONFIG(case, True)
    assert study.VARIANTS == ('MLP', 'Adaptive', 'Uniform', 'Chebyshev', 'Fourier', 'RBF', 'Wavelet')
    return dict(passed=True, budget_multiplier=4, frozen_sources=FROZEN_HASHES,
                initialization='from scratch; no pilot checkpoint loaded', optimizer_steps=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', choices=['A', 'B', 'C'], required=True)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert platform.system() == 'Linux', 'Cloud only; no local training'
    assert study.torch.cuda.is_available() and study.torch.__version__ == '2.5.1+cu121'
    protocol = protocol_test()
    hashes = digest()
    # Configure the frozen training function in memory; never edit pilot sources.
    study.config = config
    study.digest = digest
    assert study.train.__globals__['config'] is config
    assert study.train.__globals__['digest'] is digest
    root = ROOT / ('smoke_'+args.node if args.smoke else args.node)
    root.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        report = study.self_test()
        report.update(protocol=protocol, source_hashes=hashes)
        study.base.save(root/'self_test.json', report)
        print('FULL_PROTOCOL_AND_MODEL_SELF_TEST_PASSED', flush=True)
        return
    verification = json.loads((ROOT/args.node/'self_test.json').read_text(encoding='utf-8'))
    assert verification['passed'] and verification['source_hashes'] == hashes
    seed = {'A': 42, 'B': 43, 'C': 44}[args.node]
    jobs = [dict(case=case, variant=variant, seed=seed)
            for case in ('KdV', 'NS_periodic') for variant in study.VARIANTS]
    if args.smoke:
        jobs = [dict(case=case, variant='Adaptive', seed=seed) for case in ('KdV', 'NS_periodic')]
    manifest = dict(node=args.node, jobs=jobs, source_hashes=hashes,
        budgets={case:config(case, args.smoke) for case in ('KdV', 'NS_periodic')},
        purpose='Full-budget follow-up to the completed pilot; no interior labels or parameter inversion',
        initialization='From scratch with the same paired seeds; no pilot weights or optimizer state',
        architecture='Unchanged pilot: only 16-term input basis varies; hidden KAN remains Chebyshev',
        seeds=[42, 43, 44], all_seeds_reported=True,
        limitations=['Pilot-informed follow-up, not independent confirmatory evidence',
                     'Equal iteration budgets and approximate parameters, not equal wall-clock compute',
                     'Three seeds; do not discard high-error runs',
                     'Soft gates are not identifiable importance or spatial partitions',
                     'Full budget does not itself establish convergence or manuscript readiness'])
    path = root/'manifest.json'
    import fcntl
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            assert json.loads(path.read_text(encoding='utf-8')) == manifest
        else:
            study.base.save(path, manifest)
        results = []
        for job in jobs:
            try:
                result = study.train(job, root, args.smoke)
            except Exception as exc:
                study.base.save(root/'progress.json', dict(status='failed', job=job, error=repr(exc)))
                raise
            results.append(result)
            study.base.save(root/'completed.json', results)
            study.torch.cuda.empty_cache()
        study.base.save(root/'progress.json', dict(status='complete', completed=len(results), jobs=len(jobs)))


if __name__ == '__main__':
    main()
