"""CPU-only extraction of frozen observations and disjoint evaluation indexes."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'adaptive_input_basis'/'fixed_inverse_observations'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_array(path, **arrays):
    if path.exists():
        with np.load(path) as existing:
            assert set(existing.files) == set(arrays)
            assert all(np.array_equal(existing[k], v) for k, v in arrays.items())
    else:
        np.savez_compressed(path, **arrays)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    klabels = ROOT/'auxiliary_suite/KdV_Hybrid_seed43_noise0_initial1_0.005/training_labels.npz'
    kref = ROOT/'benchmark_audit/KdV_crosschecked_reference.mat'
    nsplit = ROOT/'NSdiagnostics/independent_solution_fixed_split.npz'
    nref = ROOT/'NSdiagnostics/independent_solution_diagnostic_data.npz'
    labels, ref = np.load(klabels), loadmat(kref)
    x, t = ref['x'].ravel(), ref['t'].ravel()
    xx, tt = np.meshgrid(x, t, indexing='ij')
    coordinates = np.stack([xx.ravel(), tt.ravel()], 1)
    lookup = {(float(a), float(b)):i for i, (a, b) in enumerate(coordinates.astype(np.float32))}
    z = np.concatenate([labels['x'], labels['t']], 1)
    train = np.array([lookup[(float(a), float(b))] for a, b in z], dtype=np.int64)
    test = np.flatnonzero(labels['heldout'].ravel() & (coordinates[:, -1] > 0))
    assert not np.intersect1d(train, test).size
    assert np.allclose(labels['clean'], ref['usol'].ravel()[train, None], atol=2e-6)
    assert np.array_equal(labels['clean'], labels['noisy'])
    save_array(OUT/'KdV_observations.npz', coordinates=z, target=labels['clean'], indices=train)
    save_array(OUT/'KdV_split.npz', train=train, test=test)
    counts = {'KdV':dict(observations=len(z), interior_observations=int((z[:, -1]>0).sum()),
                        unique_training_points=len(np.unique(train)), heldout=len(test))}
    labels.close()
    with np.load(nsplit) as split, np.load(nref) as ref:
        train = split['train']
        test = split['test'][ref['coordinates'][split['test'], -1] > 0]
        assert not np.intersect1d(train, test).size
        z, target = ref['coordinates'][train], ref['velocity'][train]
        save_array(OUT/'NS_periodic_observations.npz', coordinates=z, target=target, indices=train)
        save_array(OUT/'NS_periodic_split.npz', train=train, test=test)
        counts['NS_periodic'] = dict(observations=len(z), interior_observations=int((z[:, -1]>0).sum()),
                                   unique_training_points=len(np.unique(train)), heldout=len(test))
    report = dict(counts=counts, noise=0, sources={str(p.relative_to(ROOT)):sha(p) for p in [klabels,kref,nsplit,nref]},
                  files={p.name:sha(p) for p in sorted(OUT.glob('*.npz'))},
                  disjoint_train_test=True, local_optimizer_steps=0,
                  evaluation='Original heldout indexes, with initial time excluded; all-grid error is secondary')
    path = OUT/'split_checks.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == report
    else:
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
