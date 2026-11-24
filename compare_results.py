"""Compare freshly recomputed values to the packaged numerical records."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch

BASE=Path(__file__).resolve().parent
sys.path.insert(0, str(BASE/'src'))
from experiment_paths import model_directory, paths
sys.stdout.reconfigure(encoding='utf-8')


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main(out):
    config=paths()
    inputs=config.archive.parent/'03_verification_inputs'
    prior={r['id']:r for r in read(inputs/'comparison_records/all_runs.json')['runs']}
    work=out/'independent_tests'
    main_rows=[read(p) for p in sorted((work/'run_records').glob('*.json'))]
    auxiliary=read(out/'auxiliary_test_results/auxiliary_results.json')['runs']
    assert len(main_rows)==1071 and len(auxiliary)==567
    rows=[]
    for r in main_rows+auxiliary:
        expected=prior[r['id']]
        assert expected['scope']=='independent_test'
        value=r['new_test_error'] if 'new_test_error' in r else r['new_error']
        difference=abs(value-expected['field_error'])
        assert np.isclose(value,expected['field_error'],rtol=1e-10,atol=1e-8),(r['id'],difference)
        rows.append(dict(id=r['id'],recorded=expected['field_error'],recomputed=value,difference=difference))
    assert {r['id'] for r in rows}=={r['id'] for r in prior.values() if r['scope']=='independent_test'}
    wake=read(out/'wake_validation_results/wake_results.json')['runs']
    assert len(wake)==114 and {r['id'] for r in wake}=={
        r['id'] for r in prior.values() if r['eq']=='NS_wake'}
    wake_difference=0.
    for r in wake:
        expected=prior[r['id']]
        difference=abs(r['new_error']-expected['field_error'])
        assert difference<1e-8,(r['id'],difference)
        wake_difference=max(wake_difference,difference)
    old_boundary={r['id']:r for r in read(inputs/'comparison_records/boundary_checks.json')['runs']}
    boundary=read(out/'NSboundary_results/boundary_checks.json')['runs']
    assert len(boundary)==102
    boundary_difference=0.
    for r in boundary:
        old=old_boundary[r['id']]
        for metric in ('rms','maximum','coarse_rms','coarse_maximum'):
            delta=float(np.max(np.abs(np.array(r[metric])-old[metric])))
            assert np.allclose(r[metric],old[metric],atol=1e-9,rtol=1e-9),(r['id'],metric,delta)
            boundary_difference=max(boundary_difference,delta)
    references=[]
    for path in sorted((work/'test_references').glob('*.npz')):
        old=inputs/'test_plan_and_references/test_references'/path.name
        with np.load(path) as a,np.load(old) as b:
            assert np.array_equal(a['coordinates'],b['coordinates']),path.name
            difference=float(np.max(np.abs(a['reference']-b['reference'])))
            assert np.allclose(a['reference'],b['reference'],atol=1e-9,rtol=1e-9),(path.name,difference)
            references.append(dict(reference=path.stem,max_difference=difference))
    assert len(references)==11
    methods=read(out/'method_and_reference_checks/model_checks.json')
    assert methods['core_runs']==1122 and len(methods['models'])==1122
    assert not methods['source_hash_failures'] and not methods['failures']
    selection=read(out/'selection_audit.json')
    assert selection['passed'] and selection['test_based_reselection'] is False
    saved=read(out/'recomputed/verification_summary.json')
    assert saved['passed'] and saved['models']==1752
    coefficient_models=0
    coefficient_difference=0.
    for r in read(config.runs):
        if not r.get('coef'):continue
        state=torch.load(model_directory(r['run'])/'final.pt',map_location='cpu',weights_only=True)
        coefficients=state['log_coefficients'].double().exp().numpy().ravel()
        recorded=np.array(list(r['coef'].values()),dtype=np.float64)
        truth=np.array(list(r['true'].values()),dtype=np.float64)
        expected=np.array(list(r['param'].values()),dtype=np.float64)
        assert np.allclose(coefficients,recorded,atol=1e-10,rtol=1e-10),r['run']
        actual=100*np.abs(coefficients-truth)/np.abs(truth)
        delta=float(np.max(np.abs(actual-expected)))
        assert np.allclose(actual,expected,atol=1e-8,rtol=1e-10),(r['run'],delta)
        coefficient_difference=max(coefficient_difference,delta)
        coefficient_models+=1
    assert coefficient_models==sum(bool(r['coefficients']) for r in prior.values())
    with (out/'test_error_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(passed=True,recomputed_prediction_models=1752,fresh_test_models=len(rows),
                fresh_wake_validation_models=len(wake),wake_validation_only=114,
                max_wake_validation_difference_percentage_points=wake_difference,
                method_checked_models=1122,boundary_models=len(boundary),
                max_test_error_difference_percentage_points=max(r['difference'] for r in rows),
                max_boundary_statistic_difference=boundary_difference,
                parameter_models_checked_from_weights=coefficient_models,
                max_parameter_error_difference_percentage_points=coefficient_difference,
                selection_unchanged=True,reference_checks=references,
                reference_convergence_report='method_and_reference_checks/reference_checks.json',
                initial_checkpoints_checked=sum(bool(r['initial_checkpoint']) for r in methods['models']),
                note='stored initialization checkpoints checked where available.unstored initializations and stepwise sampling coordinates are not reconstructed.')
    (out/'verification_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
