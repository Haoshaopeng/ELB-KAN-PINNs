"""Explicit, audited Adam-boundary resume of two authorized OOM attempts."""
import argparse
import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time
import traceback

TASKS = (
    'matrix_inverse_AllenCahn_base_Chebyshev_HRBF_seed42',
    'matrix_inverse_AllenCahn_stress_Chebyshev_HRBF_seed44',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore(checkpoint, previous_history, spec, cfg, evidence, model, rng, drng):
    import torch
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['spec'] == spec and saved['config'] == cfg and saved['hashes'] == evidence
    assert saved['stage'] == 'Adam' and saved['step'] == cfg['adam_steps']
    assert previous_history[-1]['stage'] == 'Adam'
    assert previous_history[-1]['step'] == cfg['adam_steps']
    assert not any(row['stage'] == 'L-BFGS' for row in previous_history)
    assert all(torch.isfinite(t).all() for t in saved['model'].values())
    model.load_state_dict(saved['model'], strict=True)
    model.log_coefficients.requires_grad_(True)
    rng.set_state(saved['sampling_rng'])
    drng.set_state(saved['data_rng'])
    torch.set_rng_state(saved['cpu_rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])
    # The frozen protocol starts a new L-BFGS optimizer here, not an Adam step.
    return copy.deepcopy(previous_history), previous_history[-1]['elapsed_seconds']


def make_resume(module):
    original = ast.parse(inspect.getsource(module.train)).body[0]
    start = next(i for i,n in enumerate(original.body) if isinstance(n,ast.Expr)
                 and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)
                 and n.value.func.id == 'snapshot')
    tail = next(i for i,n in enumerate(original.body) if isinstance(n,ast.Assign)
                and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
                and n.value.func.attr == 'sample'
                and any(isinstance(x,ast.Constant) and x.value == 'lbfgs_nf' for x in ast.walk(n)))
    assert start < tail
    replacement = ast.parse('''
history, previous_seconds = _restore_resume(checkpoint, previous_history, spec, cfg, evidence, model, rng, drng)
start = time.perf_counter() - previous_seconds
''').body
    function = copy.deepcopy(original)
    function.name = 'resume_train'
    function.args.args.extend([ast.arg(arg='checkpoint'),ast.arg(arg='previous_history')])
    function.args.defaults.extend([ast.Constant(value=None),ast.Constant(value=None)])
    function.body = copy.deepcopy(original.body[:start]) + replacement + copy.deepcopy(original.body[tail:])
    assert ast.dump(ast.Module(body=function.body[-len(original.body[tail:]):],type_ignores=[])) == ast.dump(ast.Module(body=original.body[tail:],type_ignores=[]))
    namespace = dict(module.__dict__, _restore_resume=restore)
    tree = ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    exec(compile(tree, '<verified_adam_boundary_resume>', 'exec'), namespace)
    evidence = dict(source_sha256=digest(module.__file__), adapter_sha256=digest(__file__),
                    unchanged_lbfgs_tail=True,
                    lbfgs_ast_sha256=hashlib.sha256(ast.dump(ast.Module(body=original.body[tail:],type_ignores=[])).encode()).hexdigest())
    return namespace['resume_train'], evidence


def copy_snapshots(source, target):
    target.mkdir(parents=True,exist_ok=True)
    for path in source.glob('*.pt'):
        assert not path.name.startswith('LBFGS'), 'Resume must precede L-BFGS optimization'
        dest=target/path.name
        assert not dest.exists()
        shutil.copyfile(path,dest)
        assert digest(path)==digest(dest)


def equal_tree(a,b):
    import torch
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return type(a)==type(b) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b


def validate(task_id, output):
    import numpy as np
    import torch
    import remote
    import layer_pairs as layer
    import inverse_benchmarks as module
    spec=next(s for s in layer.jobs() if s['id']==task_id)
    layer.install(spec)
    output=Path(output)
    assert not output.exists(), 'Do not overwrite validation evidence'
    output.mkdir(parents=True)
    original=output/'continuous';resumed=output/'resumed'
    first=module.train(spec,original,True)
    previous=[r for r in remote.read(original/'history.json') if r['stage']!='L-BFGS']
    resumed.mkdir()
    (resumed/'snapshots').mkdir()
    for path in (original/'snapshots').glob('*.pt'):
        if not path.name.startswith('LBFGS'):shutil.copyfile(path,resumed/'snapshots'/path.name)
    resume,proof=make_resume(module)
    second=resume(spec,resumed,True,original/'snapshots/Adam_6.pt',previous)
    a=torch.load(original/'final.pt',map_location='cpu',weights_only=True)
    b=torch.load(resumed/'final.pt',map_location='cpu',weights_only=True)
    assert equal_tree(a,b), 'Continuous and resumed weights differ'
    a=torch.load(next((original/'snapshots').glob('LBFGS*.pt')),map_location='cpu',weights_only=False)
    b=torch.load(next((resumed/'snapshots').glob('LBFGS*.pt')),map_location='cpu',weights_only=False)
    assert equal_tree(a,b), 'Accepted optimizer or RNG state differs'
    assert first['closure_calls']==second['closure_calls'] and equal_tree(first['metrics'],second['metrics'])
    with np.load(original/'prediction.npz') as a,np.load(resumed/'prediction.npz') as b:
        assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
    a=remote.read(original/'history.json');b=remote.read(resumed/'history.json')
    clean=lambda rows:[{k:v for k,v in r.items() if k!='elapsed_seconds'} for r in rows]
    assert clean(a)==clean(b), 'Loss trajectory differs'
    proof.update(task=task_id,passed=True,cloud_cuda=True,smoke_only=True,
                 final_weights_bitwise_equal=True,optimizer_and_rng_bitwise_equal=True,
                 predictions_bitwise_equal=True,loss_trajectory_equal=True,
                 formal_budget_unchanged=True,closure_calls=first['closure_calls'])
    remote.save(output/'passed.json',proof)
    print(json.dumps(proof),flush=True)


def worker(task_id):
    import fcntl
    import torch
    import remote
    import layer_pairs as layer
    import inverse_benchmarks as module
    root=remote.RUN/'jobs'/task_id
    request=remote.read(root/'resume_request.json')
    assert task_id in TASKS and request['task_id']==task_id and request['authorized']
    assert request['adapter_sha256']==digest(__file__)
    assert request['ready_sha256']==digest(remote.HOME/request['ready_relative'])
    assert remote.read(remote.HOME/request['ready_relative'])['passed']
    task=next(t for t in remote.load_plan()['tasks'] if t['id']==task_id)
    assert task==request['task']
    for name,value in request['scientific_sha256'].items():assert digest(remote.HOME/name)==value
    attempt=root/'attempts/resume_01';attempt.mkdir(parents=True,exist_ok=True)
    with (attempt/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (attempt/'launched.json').exists(), 'No implicit recovery retry'
        remote.save(attempt/'launched.json',dict(identity=remote.identity(os.getpid()),time=time.time(),request=request))
        try:
            parent=root/'formal'
            checkpoint=parent/'snapshots/Adam_8000.pt'
            assert digest(checkpoint)==request['checkpoint_sha256']
            assert digest(root/'error.json')==request['error_sha256']
            assert digest(parent/'history.json')==request['history_sha256']
            assert not (root/'done.json').exists()
            previous=remote.read(parent/'history.json')
            spec=task['spec'];layer.install(spec)
            target=attempt/'formal';target.mkdir(exist_ok=False)
            copy_snapshots(parent/'snapshots',target/'snapshots')
            resume,proof=make_resume(module)
            assert proof['source_sha256']==request['scientific_sha256']['inverse_benchmarks.py']
            remote.save(attempt/'resume_provenance.json',dict(request=request,proof=proof,
                previous_attempt_retained=True,optimizer_boundary='fresh L-BFGS after Adam8000',
                no_repeated_adam_steps=True,timing_warning='Concurrent resumed duration is not isolated GPU efficiency'))
            result=resume(spec,target,False,checkpoint,previous)
            assert digest(checkpoint)==request['checkpoint_sha256']
            assert digest(root/'error.json')==request['error_sha256']
            assert digest(parent/'history.json')==request['history_sha256']
            files={str(p.relative_to(root)):digest(p) for p in target.iterdir() if p.is_file()}
            files.update({str(p.relative_to(root)):digest(p) for p in (target/'snapshots').glob('*.pt')})
            files['self_test.json']=digest(root/'self_test.json')
            files['attempts/resume_01/resume_provenance.json']=digest(attempt/'resume_provenance.json')
            assert digest(target/'final.pt')==result['weights_sha256']
            remote.save(root/'done.json',dict(task=task,files=files,time=time.time(),preflight=False,
                                             recovered_from_attempt='original_OOM_after_Adam8000'))
        except BaseException as exc:
            remote.save(attempt/'error.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['validate','worker'])
    p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.action=='validate':validate(args.task,args.output)
    else:worker(args.task)
