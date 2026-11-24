"""recompute wake validation errors from the final weights, not an independent test result."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import paths, model_directory
import evaluate_controls as auxiliary

CONFIG = paths()
test = auxiliary.test
audit = test.audit


def main(device):
    root = CONFIG.archive.parent / '03_verification_inputs'
    previous = test.read(root / 'comparison_records/all_runs.json')['runs']
    previous = {r['id']: r for r in previous if r['eq'] == 'NS_wake'}
    rows = [r for r in test.read(CONFIG.runs) if r['eq'] == 'NS_wake']
    assert len(rows) == len(previous) == 114
    reference_id = 'NS_wake_validation'
    reference_path = root / 'evaluation_data/references_and_indices' / (reference_id + '.npz')
    ref = audit.loadz(reference_path)
    assert ref['coordinates'].shape == (19586, 3)
    assert np.array_equal(ref['test_index'], np.arange(19586))
    output = CONFIG.output / 'wake_validation_results'
    output.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.perf_counter()
    for number, row in enumerate(rows, 1):
        key = test.task_id(row['eq'], row['mode'], row['reg']) + '_' + row['model'] + '_seed' + str(row['seed'])
        original = previous[key]
        dest, npz = output / (key + '.json'), output / (key + '.npz')
        if dest.exists():
            record = test.read(dest)
            assert test.sha(npz) == record['prediction_sha256'], key
            records.append(record)
            continue
        weights = model_directory(row['run']) / 'final.pt'
        assert test.sha(weights) == original['weights_sha256'], key
        model = auxiliary.make_model(row)
        assert sum(p.numel() for p in model.parameters()) == row['n'], key
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        model = model.double().to(device).eval()
        coordinates, expected = audit.load_reload_probe(row)
        probe = test.prediction(model, torch.tensor(coordinates, device=device), row['eq'])
        reload_error = float(np.max(np.abs(probe.detach().cpu().numpy() - expected)))
        assert reload_error < 1e-8, (key, reload_error)
        chunks = []
        for start in range(0, len(ref['coordinates']), 512):
            z = torch.tensor(ref['coordinates'][start:start + 512], device=device)
            value = test.prediction(model, z, row['eq'])
            chunks.append(value.detach().cpu().numpy())
            del value, z
        prediction = np.concatenate(chunks)
        error = float(100 * np.linalg.norm(prediction - ref['reference']) / np.linalg.norm(ref['reference']))
        assert abs(error - original['field_error']) < 1e-8, (key, error, original['field_error'])
        np.savez_compressed(npz, prediction=prediction, test_index=ref['test_index'],
                            reference_id=np.array(reference_id))
        record = dict(id=key, run=row['run'], eq=row['eq'], mode=row['mode'],
                      reg=row['reg'], model=row['model'], seed=row['seed'],
                      scope='wake_validation', points=len(prediction), new_error=error,
                      recorded_error=original['field_error'],
                      difference_percentage_points=abs(error - original['field_error']),
                      reload_max_difference=reload_error, weights_sha256=test.sha(weights),
                      reference_sha256=test.sha(reference_path), prediction_sha256=test.sha(npz))
        test.write_new(dest, record)
        records.append(record)
        del model, probe, chunks, prediction
        if number % 10 == 0:
            print('Wake validation', number, '/114', round(time.perf_counter() - started, 1), flush=True)
    summary = output / 'wake_results.json'
    if not summary.exists():
        test.write_new(summary, dict(scope='Validation only; no independent wake test data',
                                   runs=records, passed=True))
    print('Completed wake validation models:', len(records), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Use --device cpu.')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        main(args.device)
