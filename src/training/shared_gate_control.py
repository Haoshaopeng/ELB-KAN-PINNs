"""Pre-planned shared-learned-gate ablation; frozen forward/inverse training loops."""
import argparse
import json
from pathlib import Path
import platform

import torch
from torch import nn

import adaptive_basis as study
import forward_training as full
import inverse_training as inverse

HERE = Path(__file__).resolve().parent
BASE_MODEL = study.AdaptiveModel
OLD_DIAGNOSTIC = study.gate_diagnostic
OLD_INVERSE_EVALUATE = inverse.evaluate
FOLDERS = {'forward': 'shared_weight_forward', 'inverse': 'shared_weight_inverse'}
BASE_HASHES = {
    **full.FROZEN_HASHES,
    'forward_training.py': '8e70b85dba6a5ea5856ac3a1a51abe6724260aa82b5d9451d6bd8aabc7e0d27d',
    'inverse_training.py': '95755b9d326624be597b4541086a7c61b8a25c82ae3103ea48ce6b039827655f',
}


def digest():
    hashes = {name:study.base.sha(HERE/name) for name in BASE_HASHES}
    assert hashes == BASE_HASHES, 'Frozen implementation changed'
    hashes[Path(__file__).name] = study.base.sha(Path(__file__))
    return hashes


class SharedModel(BASE_MODEL):
    def __init__(self, case, variant):
        super().__init__(case, 'Adaptive' if variant == 'SharedLearned' else variant)
        if variant == 'SharedLearned':
            # No additional RNG draw: all nongate tensors match the paired Adaptive model.
            self.kan[0].logits = nn.Parameter(self.kan[0].logits.detach()[:1].clone())
            self.variant = variant


def diagnostic(model):
    previous = model.variant
    try:
        model.variant = 'Adaptive'
        result = OLD_DIAGNOSTIC(model)
    finally:
        model.variant = previous
    result.update(variables=['all input variables share one gate vector'], shared_across_variables=True)
    return result


def inverse_evaluate(model, folder, smoke):
    result = OLD_INVERSE_EVALUATE(model, folder, smoke)
    result['gate_diagnostic'] = dict(variables=['all input variables share one gate vector'],
        families=study.FAMILIES,weights=model.kan[0].weights().detach().cpu().tolist(),
        shared_across_variables=True,warning='Soft gates are not identifiable causal importance')
    return result


def install():
    digest()
    study.AdaptiveModel = SharedModel
    study.VARIANTS = ('SharedLearned',)
    study.config = full.config
    study.digest = digest
    study.gate_diagnostic = diagnostic
    inverse.digest = digest
    inverse.evaluate = inverse_evaluate


def self_test():
    checks = []
    for case, inputs in [('KdV', 2), ('NS_periodic', 3)]:
        for seed in (42,43,44):
            study.base.seed(seed)
            adaptive = BASE_MODEL(case,'Adaptive').cuda().double()
            study.base.seed(seed)
            shared = SharedModel(case,'SharedLearned').cuda().double()
            for name, value in adaptive.state_dict().items():
                if name != 'kan.0.logits':
                    assert torch.equal(value, shared.state_dict()[name]), (case,seed,name)
            q = torch.rand(5,inputs,device='cuda',dtype=torch.float64,requires_grad=True)*.8-.4
            assert torch.equal(adaptive.kan[0](q), shared.kan[0](q))
            gate = torch.tensor([[.2,-.3,.1,.4]],device='cuda',dtype=torch.float64)
            with torch.no_grad():
                adaptive.kan[0].logits.copy_(gate.expand(inputs,-1))
                shared.kan[0].logits.copy_(gate)
            assert torch.equal(adaptive.kan[0](q), shared.kan[0](q))
            a = adaptive.kan[0](q).square().sum()
            b = shared.kan[0](q).square().sum()
            ga = torch.autograd.grad(a,adaptive.kan[0].logits)[0].sum(0,keepdim=True)
            gb = torch.autograd.grad(b,shared.kan[0].logits)[0]
            assert torch.allclose(ga,gb,atol=1e-12,rtol=1e-10) and gb.abs().max()>1e-12
            ca=sum(p.numel() for p in adaptive.parameters())
            cs=sum(p.numel() for p in shared.parameters())
            assert ca-cs == 4*(inputs-1)
            checks.append(dict(case=case,seed=seed,paired_initialization=True,
                               tied_gate_gradient=True,adaptive_parameters=ca,shared_parameters=cs))
        q=q[:2].detach().requires_grad_()
        layer=shared.kan[0]
        def derivative(x):
            return torch.autograd.grad(layer(x).sum(),x,create_graph=True)[0]
        assert torch.autograd.gradcheck(layer,(q,),atol=1e-5,rtol=1e-3)
        assert torch.autograd.gradgradcheck(derivative,(q,),atol=1e-5,rtol=1e-3)
    forward_report=study.self_test()
    inverse_report=inverse.self_test('cuda')
    return dict(passed=True,source_hashes=digest(),optimizer_steps=0,
                initialization_checks=checks,forward=forward_report,inverse=inverse_report)


def run(mode,node,smoke=False):
    root=HERE/'adaptive_input_basis'/FOLDERS[mode]/('smoke_'+node if smoke else node)
    root.mkdir(parents=True,exist_ok=True)
    report=json.loads((HERE/'adaptive_input_basis'/'shared_weight_checks'/node/'self_test.json').read_text(encoding='utf-8'))
    assert report['passed'] and report['source_hashes']==digest()
    seed={'A':42,'B':43,'C':44}[node]
    jobs=[dict(case=case,variant='SharedLearned',seed=seed) for case in ('KdV','NS_periodic')]
    cfg=full.config if mode=='forward' else inverse.config
    train=study.train if mode=='forward' else inverse.train
    manifest=dict(node=node,mode=mode,jobs=jobs,source_hashes=digest(),
        budgets={case:cfg(case,smoke) for case in ('KdV','NS_periodic')},
        seeds=[42,43,44],all_seeds_reported=True,comparison='One 4-family gate vector shared across all input variables',
        nongate_initialization='Exactly paired with Adaptive for the same seed',
        scope='Pre-planned gate-sharing control; no change to original model, data, optimizer, or budget',
        timing='Concurrent independent workers; training_seconds is not an isolated GPU benchmark')
    if mode=='inverse':
        manifest['datasets']={case:inverse.data_hashes(case) for case in ('KdV','NS_periodic')}
    import fcntl
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'manifest.json'
        if path.exists():
            assert json.loads(path.read_text(encoding='utf-8'))==manifest
        else:
            study.base.save(path,manifest)
        results=[]
        for job in jobs:
            try:
                results.append(train(job,root,smoke))
            except Exception as exc:
                study.base.save(root/'progress.json',dict(status='failed',job=job,error=repr(exc)))
                raise
            study.base.save(root/'completed.json',results)
            torch.cuda.empty_cache()
        study.base.save(root/'progress.json',dict(status='complete',completed=2,jobs=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--node',choices=['A','B','C'],required=True)
    parser.add_argument('--mode',choices=['forward','inverse'],default='forward')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    assert platform.system()=='Linux', 'Cloud only'
    assert torch.cuda.is_available() and torch.__version__=='2.5.1+cu121'
    install()
    if args.self_test:
        test_root=HERE/'adaptive_input_basis'/'shared_weight_checks'/args.node
        test_root.mkdir(parents=True,exist_ok=True)
        study.base.save(test_root/'self_test.json',self_test())
        print('SHARED_GATE_CONTROL_SELF_TEST_PASSED',flush=True)
    else:
        run(args.mode,args.node,args.smoke)
