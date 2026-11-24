"""Finite missing single-family layer pairs; existing scientific modules stay frozen."""
import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
import pde_benchmarks as new
import inverse_benchmarks as new_inverse
import experiment_conditions as common
import cylinder_wake as wake
import forward_training as full
from architecture_controls import hidden_layer

HERE=Path(__file__).resolve().parent
FAMILIES=('Chebyshev','Fourier','RBF','Wavelet')
EXISTING={(f,'Chebyshev') for f in FAMILIES}|{('Fourier','Fourier')}
INSTALLED=None
SOURCE_FACTORY=None
MODULE=None


def conditions():
    result=[('forward',c,'baseline','old') for c in ('Burgers','NS_periodic')]
    result += [('inverse',c,'baseline','old') for c in ('KdV','NS_periodic')]
    result += [('inverse','NS_wake','clean','wake')]
    result += [(mode,c,r,'new') for mode in ('forward','inverse') for c in new.CASES for r in ('base','stress')]
    return result


def jobs():
    result=[]
    for mode,case,regime,protocol in conditions():
        for first in FAMILIES:
            for hidden in FAMILIES:
                if (first,hidden) in EXISTING:continue
                for seed in (42,43,44):
                    variant=first+'_H'+hidden
                    result.append(dict(id=f'matrix_{mode}_{case}_{regime}_{variant}_seed{seed}',
                        kind='layerpair',mode=mode,protocol=protocol,case=case,regime=regime,
                        condition=f'matrix_{mode}_{regime}',input_variant=first,hidden_basis=hidden,
                        variant=variant,seed=seed,job=dict(case=case,variant=variant,seed=seed)))
    assert len(result)==len({s['id'] for s in result})==693
    assert sum(s['mode']=='forward' for s in result)==330
    return result


def install(spec):
    global INSTALLED,SOURCE_FACTORY,MODULE
    assert INSTALLED is None and spec in jobs(), 'One frozen adapter per worker'
    INSTALLED=spec
    first,hidden=spec['input_variant'],spec['hidden_basis']
    if spec['protocol']=='new':
        SOURCE_FACTORY=new.Model
        original=SOURCE_FACTORY
        class Model(original):
            def __init__(self,job):
                super().__init__(dict(job,variant=first))
                # The coefficient tensors and RNG consumption match the frozen model.
                self.kan[2].family=hidden
                self.job=job
        new.Model=Model
        MODULE=new if spec['mode']=='forward' else new_inverse
        old_config=MODULE.config
        MODULE.config=lambda job,smoke=False:dict(old_config(job,smoke),layer_pair=spec)
        if spec['mode']=='inverse':
            original_hashes=new_inverse.hashes
            def hashes(job):
                value=original_hashes(job)
                value['source'][Path(__file__).name]=new.sha(__file__)
                return value
            new_inverse.hashes=hashes
    else:
        study,inverse=common.study,common.inverse
        if spec['protocol']=='wake':
            base_spec=next(s for s in wake.jobs() if s['condition']=='clean' and
                s['job']['variant']==first and s['job']['seed']==spec['seed'])
            wake.install(base_spec)
        elif spec['case']=='Burgers':
            common.burgers.install()
        elif spec['mode']=='forward':
            full.protocol_test()
            study.config,study.digest=full.config,full.digest
        SOURCE_FACTORY=study.AdaptiveModel
        original=SOURCE_FACTORY
        class Model(original):
            def __init__(self,case,variant):
                super().__init__(case,first)
                self.kan[2]=hidden_layer(self.kan[2],hidden)
                self.variant=variant
        study.AdaptiveModel=Model
        study.gate_diagnostic=lambda model:None
        MODULE=study if spec['mode']=='forward' else inverse
        old_config,old_digest=MODULE.config,MODULE.digest
        hashes=dict(old_digest());hashes[Path(__file__).name]=new.sha(__file__)
        hashes['architecture_controls.py']=new.sha(HERE/'architecture_controls.py')
        MODULE.config=lambda case,smoke=False:dict(old_config(case,smoke),layer_pair=spec)
        MODULE.digest=lambda:dict(hashes)
        common.INSTALLED=spec
    return MODULE


def model_factory(spec):
    assert INSTALLED==spec
    if spec['protocol']=='new':return MODULE.Model(spec)
    factory=common.study.AdaptiveModel if spec['mode']=='forward' else common.inverse.InverseModel
    return factory(spec['case'],spec['variant'])


def self_test(spec,device='cpu'):
    assert INSTALLED==spec
    torch.set_num_threads(1);common.study.base.seed(spec['seed'])
    model=model_factory(spec).to(device).double()
    network=model.network if spec['mode']=='inverse' else model
    common.study.base.seed(spec['seed'])
    reference=(SOURCE_FACTORY(dict(spec,variant=spec['input_variant'])) if spec['protocol']=='new'
        else SOURCE_FACTORY(spec['case'],spec['input_variant'])).to(device).double()
    before=dict(reference.named_parameters());after=dict(network.named_parameters())
    assert before.keys()==after.keys()
    assert all(torch.equal(value,after[key]) for key,value in before.items()), 'Unrelated initialization changed'
    cfg=MODULE.config(spec if spec['protocol']=='new' else spec['case'])
    assert cfg['layer_pair']==spec
    if spec['protocol']=='new':
        report=MODULE.self_test(spec,device)
    elif spec['protocol']=='wake':
        report=wake.self_test(spec,device)
    else:
        with torch.no_grad():
            for p in network.parameters():p.add_(.003*torch.randn_like(p))
        dim=2 if spec['case'] in ('Burgers','KdV') else 3
        z=torch.rand(5,dim,device=device,dtype=torch.float64).requires_grad_()
        scale=torch.linspace(.4,.9,dim,device=device,dtype=torch.float64)
        q=((z.detach()-.17)/scale).requires_grad_();errors=[]
        derivative=common.study.base.derivative
        for channel in range(model(z).shape[1]):
            for axis in range(dim):
                a,b=model(z)[:,channel:channel+1],model(q*scale+.17)[:,channel:channel+1]
                for _ in range(3):
                    a=derivative(a,z)[:,axis:axis+1];b=derivative(b,q)[:,axis:axis+1]/scale[axis]
                    assert torch.allclose(a,b,rtol=1e-7,atol=1e-8)
                    errors.append(float((a-b).abs().max().detach()))
        if spec['mode']=='inverse':residual=common.inverse.residual(model,z)
        elif spec['case']=='Burgers':residual=common.burgers.residual(model,z)
        else:
            # Reuse the unchanged physical forward loss and boundary operator.
            residual=common.study.base.loss_parts(model,z,torch.rand(4,2,device=device,dtype=torch.float64))[0]
        loss=residual.square().mean();assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if spec['mode']=='inverse':
            assert model.log_coefficients.grad is not None and torch.all(model.log_coefficients.grad.abs()>1e-14)
            common.inverse.load_observations(spec['case'],device)
        restored=model_factory(spec).to(device).double();restored.load_state_dict(model.state_dict())
        assert torch.equal(model(z),restored(z))
        report=dict(passed=True,physical_derivative_max_difference=max(errors),finite_backward=True,reload_exact=True)
    return dict(report,spec=spec,optimizer_steps=0,device=device,paired_initial_tensors=True,
        config=cfg,source_sha256=new.sha(__file__))


def train_new_forward(spec,root,smoke=False):
    """Frozen forward optimizer sequence, with non-mutating trajectory snapshots."""
    assert platform.system()=='Linux' and torch.cuda.is_available(), 'Cloud only'
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    assert not (root/'started.json').exists(), 'No implicit retry'
    cfg=new.config(spec,smoke);torch.set_num_threads(1)
    torch.manual_seed(spec['seed']);torch.cuda.manual_seed_all(spec['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=new.Model(spec).cuda();rng=torch.Generator(device='cuda').manual_seed(spec['seed']+1000)
    new.save(root/'started.json',dict(job=spec,config=cfg,source_sha256=new.sha(__file__),time=time.time()))
    snapshots=root/'snapshots';snapshots.mkdir(exist_ok=True)
    history=[];start=time.time()
    def snapshot(stage,step,opt=None,scheduler=None):
        path=snapshots/f'{stage}_{step}.pt';assert not path.exists()
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict() if opt else None,
            scheduler=scheduler.state_dict() if scheduler else None,rng=rng.get_state(),
            torch_cpu_rng=torch.get_rng_state(),torch_cuda_rng=torch.cuda.get_rng_state_all(),
            spec=spec,config=cfg,stage=stage,step=step),path)
    def record(stage,step,loss):
        row=dict(stage=stage,step=step,loss=float(loss.detach()),elapsed_seconds=time.time()-start)
        history.append(row);new.save(root/'history.json',history)
        new.save(root/'progress.json',dict(job=spec,config=cfg,**row))
    snapshot('initial',0)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
    scheduler=torch.optim.lr_scheduler.StepLR(opt,max(1,cfg['adam_steps']//2),gamma=.5)
    for step in range(1,cfg['adam_steps']+1):
        z=new.sample(spec,cfg['nf'],rng,'cuda');opt.zero_grad(set_to_none=True)
        loss=new.residual(model,z).square().mean();assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();scheduler.step()
        if step==1 or step%100==0 or step==cfg['adam_steps']:record('Adam',step,loss)
        if step%500==0 or step==cfg['adam_steps']:snapshot('Adam',step,opt,scheduler)
    z=new.sample(spec,cfg['lbfgs_nf'],rng,'cuda')
    opt=torch.optim.LBFGS(model.parameters(),lr=.8,max_iter=cfg['lbfgs_max_iter'],max_eval=2*cfg['lbfgs_max_iter'],
        history_size=50,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-10)
    calls=0
    def closure():
        nonlocal calls
        opt.zero_grad(set_to_none=True);loss=new.residual(model,z).square().mean();assert torch.isfinite(loss)
        loss.backward();calls+=1
        if calls==1 or calls%25==0:record('L-BFGS',calls,loss)
        return loss
    assert float(closure().detach())==float(closure().detach())
    opt.step(closure);snapshot('LBFGS_accepted_final',int(next(iter(opt.state.values())).get('n_iter',0)),opt)
    torch.save(model.state_dict(),root/'final.pt');torch.cuda.synchronize()
    result=dict(job=spec,config=cfg,weights_sha256=new.sha(root/'final.pt'),source_sha256=new.sha(__file__),
        parameters=sum(p.numel() for p in model.parameters()),training_seconds=time.time()-start,
        closure_calls=calls,timing_warning='Concurrent duration is not isolated GPU efficiency',
        metrics=new.evaluate(model,root,spec))
    new.save(root/'result.json',result)
    new.save(root/'snapshot_manifest.json',dict(files={str(p.relative_to(root)):new.sha(p) for p in snapshots.glob('*.pt')}))
    return result


def train(spec,root,smoke=False):
    assert INSTALLED==spec and platform.system()=='Linux' and torch.cuda.is_available()
    if spec['protocol']=='new':
        return train_new_forward(spec,root,smoke) if spec['mode']=='forward' else new_inverse.train(spec,root,smoke)
    return common.train(spec,Path(root),smoke)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    spec=next(s for s in jobs() if s['id']==a.task);install(spec)
    new.save(a.output,self_test(spec,a.device))
