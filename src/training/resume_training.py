"""Authorized Adam-boundary continuation; frozen scientific tails are reused verbatim."""
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

import resume_inverse_training as previous

TASKS = {
    'matrix_inverse_Heat2D_stress_Chebyshev_HRBF_seed44': 'N5',
    'matrix_forward_Helmholtz2D_base_Fourier_HWavelet_seed44': 'N6',
    'matrix_forward_Burgers_baseline_Fourier_HWavelet_seed42': 'N6',
}
digest = previous.digest


def result_folder(spec, root):
    root = Path(root)
    return root if spec['protocol'] == 'new' else root / (
        spec['case'] + '_' + spec['variant'] + '_seed' + str(spec['seed']))


def restore_forward(checkpoint, history, spec, cfg, model, rng, hashes=None):
    import numpy as np
    import random
    import torch
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['config'] == cfg
    assert saved['stage'] == 'Adam' and saved['step'] == cfg['adam_steps']
    assert history[-1]['stage'] == 'Adam' and history[-1]['step'] == cfg['adam_steps']
    assert not any(row['stage'] == 'L-BFGS' for row in history)
    assert all(torch.isfinite(t).all() for t in saved['model'].values())
    assert all(model.state_dict()[k].dtype == v.dtype and model.state_dict()[k].shape == v.shape
               for k, v in saved['model'].items()), 'Checkpoint precision or shape mismatch'
    if hashes is None:
        assert saved['spec'] == spec
        rng.set_state(saved['rng'])
    else:
        assert saved['job'] == spec and saved['source_hashes'] == hashes
        rng.set_state(saved['sampling_rng'])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
    model.load_state_dict(saved['model'], strict=True)
    torch.set_rng_state(saved['torch_cpu_rng'])
    torch.cuda.set_rng_state_all(saved['torch_cuda_rng'])
    return copy.deepcopy(history), history[-1]['elapsed_seconds']


def make_forward_resume(module, old=False):
    source = module.train if old else module.train_new_forward
    original = ast.parse(inspect.getsource(source)).body[0]
    if old:
        start = next(i for i, n in enumerate(original.body) if isinstance(n, ast.For))
        tail = start + 1
        replacement = '''
history, prior_seconds = _restore(checkpoint_path, previous_history, job, cfg, model, rng, hashes)
start = time.perf_counter() - prior_seconds
'''
    else:
        start = next(i for i, n in enumerate(original.body) if isinstance(n, ast.Expr)
                     and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                     and n.value.func.id == 'snapshot')
        tail = next(i for i, n in enumerate(original.body) if isinstance(n, ast.Assign)
                    and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                    and n.value.func.attr == 'sample'
                    and any(isinstance(x, ast.Constant) and x.value == 'lbfgs_nf' for x in ast.walk(n)))
        replacement = '''
history, prior_seconds = _restore(checkpoint_path, previous_history, spec, cfg, model, rng)
start = time.time() - prior_seconds
'''
    assert start < tail
    fn = copy.deepcopy(original)
    fn.name = 'resume_train'
    fn.args.args.extend([ast.arg(arg='checkpoint_path'), ast.arg(arg='previous_history')])
    fn.args.defaults.extend([ast.Constant(None), ast.Constant(None)])
    fn.body = copy.deepcopy(original.body[:start]) + ast.parse(replacement).body + copy.deepcopy(original.body[tail:])
    unchanged = ast.dump(ast.Module(body=original.body[tail:], type_ignores=[]))
    assert ast.dump(ast.Module(body=fn.body[-len(original.body[tail:]):], type_ignores=[])) == unchanged
    namespace = dict(module.__dict__, _restore=restore_forward)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 '<verified_forward_resume>', 'exec'), namespace)
    return namespace['resume_train'], dict(source_sha256=digest(module.__file__),
        unchanged_lbfgs_tail=True, lbfgs_ast_sha256=hashlib.sha256(unchanged.encode()).hexdigest())


def archival_wrapper(common):
    # Only suppress re-creation of the initial snapshot; original files are copied intact.
    tree = ast.parse(inspect.getsource(common.train))
    factory = next(n for n in tree.body[0].body if isinstance(n, ast.FunctionDef) and n.name == 'factory')
    start = next(i for i, n in enumerate(factory.body) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'path')
    assert len(factory.body[start:-1]) == 3 and isinstance(factory.body[-1], ast.Return)
    factory.body = factory.body[:start] + factory.body[-1:]
    namespace = dict(common.__dict__)
    exec(compile(ast.fix_missing_locations(tree), '<archive_original_snapshots>', 'exec'), namespace)
    return namespace['train']


def run(spec, root, smoke=False, checkpoint=None, history=None):
    import torch
    import layer_pairs as layer
    if spec['protocol'] == 'new':
        if spec['mode'] == 'inverse':
            module = layer.new_inverse
            fn, proof = previous.make_resume(module)
            plain = module.train
        else:
            fn, proof = make_forward_resume(layer)
            plain = layer.train_new_forward
        return (fn(spec, root, smoke, checkpoint, history) if checkpoint else plain(spec, root, smoke)), proof
    common = layer.common
    module = common.study
    original_train, factory = module.train, module.AdaptiveModel
    checkpoint_fn, lbfgs = module.checkpoint, torch.optim.LBFGS
    fn, proof = make_forward_resume(module, old=True)
    try:
        if checkpoint:
            def resumed_train(job, target, short=False):
                fn.__globals__['AdaptiveModel'] = module.AdaptiveModel
                return fn(job, target, short, checkpoint, history)
            module.train = resumed_train
            result = archival_wrapper(common)(spec, Path(root), smoke)
        else:
            result = common.train(spec, Path(root), smoke)
        return result, proof
    finally:
        module.train, module.AdaptiveModel = original_train, factory
        module.checkpoint, torch.optim.LBFGS = checkpoint_fn, lbfgs


def equal(a, b):
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) == type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def copy_prior(source, target):
    target.mkdir(parents=True, exist_ok=True)
    for path in source.glob('*.pt'):
        if path.name.startswith('LBFGS'):
            continue
        dest = target / path.name
        assert not dest.exists()
        shutil.copyfile(path, dest)
        assert digest(path) == digest(dest)


def validate(task_id, output):
    import numpy as np
    import torch
    import remote
    import layer_pairs as layer
    spec = next(s for s in layer.jobs() if s['id'] == task_id)
    assert torch.cuda.is_available() and torch.__version__ == '2.5.1+cu121'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    module = layer.install(spec)
    assert layer.self_test(spec, 'cuda')['passed']
    output = Path(output)
    assert not output.exists(), 'Validation evidence must not be overwritten'
    output.mkdir(parents=True)
    original, resumed = output / 'continuous', output / 'resumed'
    first, proof = run(spec, original, True)
    aroot, broot = result_folder(spec, original), result_folder(spec, resumed)
    history = [r for r in remote.read(aroot / 'history.json') if r['stage'] != 'L-BFGS']
    step = first['config']['adam_steps']
    copy_prior(aroot / 'snapshots', broot / 'snapshots')
    second, _ = run(spec, resumed, True, aroot / 'snapshots' / f'Adam_{step}.pt', history)
    load = lambda p: torch.load(p, map_location='cpu', weights_only=False)
    assert equal(load(aroot / 'final.pt'), load(broot / 'final.pt')), 'Final weights differ'
    assert equal(load(next((aroot / 'snapshots').glob('LBFGS*.pt'))),
                 load(next((broot / 'snapshots').glob('LBFGS*.pt')))), 'Optimizer/RNG differ'
    assert first['closure_calls'] == second['closure_calls']
    predictions = list(aroot.glob('*.npz'))
    assert predictions
    for path in predictions:
        with np.load(path) as a, np.load(broot / path.name) as b:
            assert a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
    clean = lambda rows: [{k: v for k, v in row.items() if k != 'elapsed_seconds'} for row in rows]
    assert clean(remote.read(aroot / 'history.json')) == clean(remote.read(broot / 'history.json'))
    if 'metrics' in first:
        assert equal(first['metrics'], second['metrics'])
    proof.update(task=task_id, passed=True, adapter_sha256=digest(__file__),
        support_adapter_sha256=digest(previous.__file__), cloud_cuda=True, smoke_only=True,
        final_weights_bitwise_equal=True, optimizer_and_rng_bitwise_equal=True,
        predictions_bitwise_equal=True, loss_trajectory_equal=True,
        formal_budget_unchanged=True, closure_calls=first['closure_calls'])
    remote.save(output / 'passed.json', proof)
    print(json.dumps(proof), flush=True)


def worker(task_id):
    import fcntl
    import torch
    import remote
    import layer_pairs as layer
    assert torch.cuda.is_available() and torch.__version__ == '2.5.1+cu121'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    root = remote.RUN / 'jobs' / task_id
    task = next(t for t in remote.load_plan()['tasks'] if t['id'] == task_id)
    req = remote.recovery_request(root, task)
    assert task_id in TASKS and req['adapter'] == Path(__file__).name
    assert req['adapter_sha256'] == digest(__file__)
    assert req['support_adapter_sha256'] == digest(previous.__file__)
    ready = remote.read(remote.HOME / req['ready_relative'])
    assert any(r['task'] == task_id and r['passed'] for r in ready['reports'])
    for name, value in req['scientific_sha256'].items():
        assert digest(remote.HOME / name) == value
    attempt = root / 'attempts/resume_01'
    attempt.mkdir(parents=True, exist_ok=True)
    with (attempt / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (attempt / 'launched.json').exists(), 'No implicit retry'
        remote.save(attempt / 'launched.json', dict(identity=remote.identity(os.getpid()), time=time.time(), request=req))
        try:
            assert not (root / 'done.json').exists()
            spec = task['spec']
            layer.install(spec)
            parent = result_folder(spec, root / 'formal')
            checkpoint = root / req['checkpoint_relative']
            history_path = root / req['history_relative']
            assert checkpoint == parent / 'snapshots/Adam_8000.pt'
            assert history_path == parent / 'history.json'
            target = attempt / 'formal'
            assert not target.exists()
            out = result_folder(spec, target)
            snapshots = list((parent / 'snapshots').glob('*.pt'))
            assert snapshots and not any(p.name.startswith('LBFGS') for p in snapshots)
            before = {str(p.relative_to(root)): digest(p) for p in snapshots + [history_path, root / 'error.json']}
            copy_prior(parent / 'snapshots', out / 'snapshots')
            remote.save(attempt / 'resume_provenance.json', dict(request=req,
                previous_attempt_retained=True, optimizer_boundary='fresh L-BFGS after Adam8000',
                no_repeated_adam_steps=True, original_files_sha256=before,
                timing_warning='Concurrent/resumed duration is not isolated GPU efficiency'))
            result, proof = run(spec, target, False, checkpoint, remote.read(history_path))
            assert all(digest(root / p) == value for p, value in before.items())
            assert digest(out / 'final.pt') == result['weights_sha256']
            assert all(digest(remote.HOME / p) == value for p, value in req['scientific_sha256'].items())
            provenance = remote.read(attempt / 'resume_provenance.json')
            provenance['proof'] = proof
            remote.save(attempt / 'resume_provenance.json', provenance)
            files = {str(p.relative_to(root)): digest(p) for p in target.rglob('*') if p.is_file()}
            files['self_test.json'] = digest(root / 'self_test.json')
            files['attempts/resume_01/resume_provenance.json'] = digest(attempt / 'resume_provenance.json')
            remote.save(root / 'done.json', dict(task=task, files=files, time=time.time(), preflight=False,
                recovered_from_attempt='original_OOM_after_Adam8000'))
        except BaseException as exc:
            remote.save(attempt / 'error.json', dict(error=repr(exc), traceback=traceback.format_exc(), time=time.time()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['validate', 'worker'])
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'validate':
        validate(args.task, args.output)
    else:
        worker(args.task)
