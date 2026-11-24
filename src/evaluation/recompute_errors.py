"""Recompute errors from stored predictions, boundary data and indices using NumPy."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def regenerated_prediction(run_dir, key):
    options = [
        (run_dir/'independent_tests/predictions'/(key+'.npz'),
         run_dir/'independent_tests/run_records'/(key+'.json')),
        (run_dir/'auxiliary_test_results'/(key+'.npz'), run_dir/'auxiliary_test_results'/(key+'.json')),
        (run_dir/'wake_validation_results'/(key+'.npz'), run_dir/'wake_validation_results'/(key+'.json')),
    ]
    found = [(p, r) for p, r in options if p.exists() and r.exists()]
    if len(found) != 1:
        raise FileNotFoundError('missing or duplicate prediction, run the full verification first: '+key)
    path, record_path = found[0]
    record = read(record_path)
    digest=record.get('prediction_file_sha256',record.get('prediction_sha256'))
    assert digest and hashlib.sha256(path.read_bytes()).hexdigest()==digest, key
    return path


def main(root, output=None, run_dir=None):
    required=[root/'comparison_records/all_runs.json',root/'comparison_records/boundary_checks.json',
              root/'evaluation_data/references_and_indices',root/'evaluation_data/training_and_validation_indices',
              root/'evaluation_data/periodic_boundaries']
    missing=[str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing verification inputs: \n'+'\n'.join(missing)+
                                '\nspecify the verification directory containing evaluation inputs and comparison records, not the code directory.')
    records=read(root/'comparison_records/all_runs.json')
    references={}
    rows=[]
    max_field_diff=0.
    max_parameter_diff=0.
    for r in records['runs']:
        if run_dir:
            path=regenerated_prediction(run_dir,r['id'])
        else:
            path=root/r['prediction_file']
            if not path.exists():
                raise FileNotFoundError('the reduced archive does not include the full prediction arrays.run first“run_verification.py”, or use --run-dir specify the completed run directory.')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==r['prediction_sha256'],r['id']
        if r['reference_id'] not in references:
            with np.load(root/'evaluation_data/references_and_indices'/(r['reference_id']+'.npz'),allow_pickle=False) as a:
                references[r['reference_id']]={k:a[k] for k in a.files}
        reference=references[r['reference_id']]
        with np.load(path,allow_pickle=False) as a:
            prediction=a['prediction'].astype(np.float64)
            assert str(a['reference_id'])==r['reference_id']
            assert np.array_equal(a['test_index'],np.arange(len(reference['reference'])))
        target=reference['reference'].astype(np.float64)
        assert prediction.shape==target.shape and len(target)==r['points']
        assert np.isfinite(prediction).all() and np.isfinite(target).all()
        eu=float(100*np.sqrt(np.sum((prediction-target)**2)/np.sum(target**2)))
        diff=abs(eu-r['field_error']);max_field_diff=max(max_field_diff,diff)
        assert diff<1e-9,r['id']
        for name,value in r['coefficients'].items():
            truth=r['coefficient_truth'][name]
            error=100*abs(value-truth)/abs(truth)
            delta=abs(error-r['coefficient_errors'][name]);max_parameter_diff=max(max_parameter_diff,delta)
            assert delta<1e-8,(r['id'],name)
        rows.append(dict(id=r['id'],scope=r['scope'],recomputed_error_percent=eu,recorded_error_percent=r['field_error'],difference_percentage_points=diff))
    boundary=read(root/'comparison_records/boundary_checks.json')
    max_boundary_diff=0.
    for r in boundary['runs']:
        if run_dir:
            path=run_dir/'NSboundary_results'/(r['id']+'.npz')
            fresh=read(run_dir/'NSboundary_results'/(r['id']+'.json'))
            assert hashlib.sha256(path.read_bytes()).hexdigest()==fresh['mismatch_file_sha256']
        else:
            path=root/'evaluation_data/periodic_boundaries'/(r['id']+'.npz')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==r['mismatch_file_sha256']
        a=np.load(path,allow_pickle=False)['boundary_difference']
        assert a.shape==(2,16705,6)
        rms=np.sqrt(np.mean(a*a,axis=(0,1)))
        maximum=np.max(np.abs(a),axis=(0,1))
        diff=max(float(np.max(abs(rms-r['rms']))),float(np.max(abs(maximum-r['maximum']))))
        max_boundary_diff=max(max_boundary_diff,diff)
        assert diff<1e-10,r['id']
    index_files=list((root/'evaluation_data/training_and_validation_indices').glob('*.npz'))
    assert index_files, 'missing from the training and validation index directory NPZ file'
    for p in index_files:
        a=np.load(p,allow_pickle=False)
        assert len(np.intersect1d(a['train_index'],a['validation_index']))==0,p.name
        assert a['train_index'].max()<len(a['reference_coordinates'])
        assert a['validation_index'].max()<len(a['reference_coordinates'])
    result=dict(models=len(rows),new_test=sum(r['scope']=='independent_test' for r in rows),wake_validation=sum(r['scope']=='wake_validation' for r in rows),boundary_models=len(boundary['runs']),max_field_difference_percentage_points=max_field_diff,max_parameter_difference_percentage_points=max_parameter_diff,max_boundary_difference=max_boundary_diff,prediction_source='regenerated_from_final_weights' if run_dir else 'saved_arrays',passed=True)
    out=output or root/'recomputed';out.mkdir(parents=True,exist_ok=True)
    (out/'verification_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'per_run_recomputation.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from experiment_paths import paths
    config=paths()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path,help='verification_inputs_dir, containing evaluation inputs and comparison records')
    parser.add_argument('--run-dir',type=Path,help='directory containing this verification run, includes regenerated predictions')
    args=parser.parse_args()
    main(args.root.expanduser().resolve(),config.output/'recomputed',
         args.run_dir.expanduser().resolve() if args.run_dir else None)
