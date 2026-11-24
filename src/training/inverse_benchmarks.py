"""Observed-data inverse companions to the four frozen new-PDE forward cases."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import numpy as np
import torch
from torch import nn
import pde_benchmarks as forward

HERE=Path(__file__).resolve().parent
DATA=HERE/'new_inverse_data'
INITIAL={'Heat2D':[.05,.02],'AllenCahn':[.002,2.],'Helmholtz2D':[100.],'Wave1D':[2.25]}
NAMES={'Heat2D':['ax','ay'],'AllenCahn':['epsilon','reaction'],'Helmholtz2D':['k_squared'],'Wave1D':['c_squared']}

def jobs():
    return [dict(id=f'inverse_new_{c}_{r}_{v}_seed{s}',kind='newinverse',case=c,regime=r,
                 condition=r+'_inverse',variant=v,seed=s,mode='inverse')
            for c in forward.CASES for r in ('base','stress') for v in forward.MODELS for s in (42,43,44)]

def prefix(spec):return spec['case']+'_'+spec['regime']

def reference_coefficients(spec):
    p=forward.physics(spec['case'],spec['regime']);case=spec['case']
    return np.array([p['ax'],p['ay']] if case=='Heat2D' else [p['epsilon'],p['reaction']] if case=='AllenCahn'
                    else [p['k']**2] if case=='Helmholtz2D' else [p['c']**2])

def prepare_data():
    DATA.mkdir(exist_ok=True);reports=[]
    for case in forward.CASES:
        for regime in ('base','stress'):
            stem=case+'_'+regime;source=forward.DATA/(stem+'.npz')
            with np.load(source) as data:z,y=data['coordinates'],data['target']
            lower,upper=z.min(0),z.max(0)
            spatial=2 if case in ('Heat2D','Helmholtz2D') else 1
            eligible=np.flatnonzero(np.all((z[:,:spatial]>lower[:spatial])&(z[:,:spatial]<upper[:spatial]),axis=1))
            rng=np.random.default_rng(20261018+100*forward.CASES.index(case)+(1 if regime=='stress' else 0))
            train=np.sort(rng.choice(eligible,800,replace=False));test=np.setdiff1d(np.arange(len(z)),train)
            assert not np.intersect1d(train,test).size
            target=DATA/(stem+'_observations.npz');split=DATA/(stem+'_split.npz')
            if target.exists():
                with np.load(target) as old:assert np.array_equal(old['indices'],train) and np.array_equal(old['target'],y[train])
            else:np.savez_compressed(target,coordinates=z[train],target=y[train],indices=train)
            if split.exists():
                with np.load(split) as old:assert np.array_equal(old['train'],train) and np.array_equal(old['test'],test)
            else:np.savez_compressed(split,train=train,test=test)
            meta=dict(case=case,regime=regime,reference_sha256=forward.sha(source),
                observations_sha256=forward.sha(target),split_sha256=forward.sha(split),
                observation_count=800,test_count=len(test),no_overlap=True,
                splitting='Fixed across all models and all training seeds; never selected by test error')
            forward.save(DATA/(stem+'_metadata.json'),meta);reports.append(meta)
    return reports

def config(spec,smoke=False):
    return dict(pretrain_steps=3 if smoke else 3000,adam_steps=6 if smoke else 8000,
        nf=16 if smoke else 1024,lbfgs_nf=32 if smoke else 2048,batch=32 if smoke else 800,
        lbfgs_max_iter=3 if smoke else 400,network_lr=5e-4,coefficient_lr=2e-3,
        initial_unknown_coefficients=INITIAL[spec['case']],unknown_parameters=NAMES[spec['case']],
        precision='FP32; TF32 off',evaluation_precision='FP64',hard_initial_boundary=True,
        loss='normalized observed-field MSE + physical PDE residual MSE',truth_prior=False,
        forcing='Helmholtz fixed source function; independent of estimated k_squared',
        checkpoint='final accepted weights; no test-based selection',smoke=smoke)

class Model(nn.Module):
    def __init__(self,spec):
        super().__init__();self.spec=spec;self.case=spec['case']
        self.network=forward.Model(spec)
        # Only prescribed initial/boundary modes remain in the predictor.
        for key in ('ax','ay','epsilon','reaction','k','c'):self.network.p.pop(key,None)
        self.log_coefficients=nn.Parameter(torch.log(torch.tensor(INITIAL[self.case],dtype=torch.float32)))
        self.source_modes=[]
        if self.case=='Helmholtz2D':
            p=forward.physics(self.case,spec['regime'])
            self.source_modes=[(p['m'],p['n'],math.pi**2*(p['m']**2+p['n']**2)-p['k']**2),
                (p['m']+1,p['n']+2,.25*(math.pi**2*((p['m']+1)**2+(p['n']+2)**2)-p['k']**2))]
        scale=max(1.,math.sqrt(sum(a*a for _,_,a in self.source_modes)/4))
        self.register_buffer('residual_scale',torch.tensor(scale,dtype=torch.float32))
    @property
    def coefficients(self):return self.log_coefficients.exp()
    def forward(self,z):return self.network(z)
    def forcing(self,z):
        return sum(a*torch.sin(m*math.pi*z[:,:1])*torch.sin(n*math.pi*z[:,1:2]) for m,n,a in self.source_modes)

def residual(model,z):
    u=model(z);d=forward.grad(u,z);xx=forward.grad(d[:,:1],z)[:,:1];a=model.coefficients
    if model.case=='AllenCahn':return d[:,1:2]-a[0]*xx+a[1]*(u**3-u)
    if model.case=='Wave1D':return forward.grad(d[:,1:2],z)[:,1:2]-a[0]*xx
    yy=forward.grad(d[:,1:2],z)[:,1:2]
    if model.case=='Heat2D':return d[:,2:3]-a[0]*xx-a[1]*yy
    return (-xx-yy-a[0]*u-model.forcing(z))/model.residual_scale

def load_observations(spec,device):
    stem=prefix(spec)
    with np.load(DATA/(stem+'_observations.npz')) as obs,np.load(DATA/(stem+'_split.npz')) as split:
        assert np.array_equal(obs['indices'],split['train']) and not np.intersect1d(obs['indices'],split['test']).size
        z=torch.tensor(obs['coordinates'],device=device,dtype=torch.float32)
        y=torch.tensor(obs['target'],device=device,dtype=torch.float32)
    return z,y,y.square().mean().detach()

def hashes(spec):
    return dict(source={n:forward.sha(HERE/n) for n in (Path(__file__).name,'pde_benchmarks.py','architecture_controls.py')},
        data={suffix:forward.sha(DATA/(prefix(spec)+suffix)) for suffix in ('_observations.npz','_split.npz','_metadata.json')})

def self_test(spec,device='cpu'):
    torch.set_num_threads(1);torch.manual_seed(spec['seed'])
    model=Model(spec).to(device).double()
    class Polynomial(nn.Module):
        def __init__(self):
            super().__init__();self.case=model.case;self.coefficients=model.coefficients
            self.residual_scale=model.residual_scale;self.forcing=model.forcing
        def forward(self,z):return z.square().sum(1,keepdim=True)
    probe=torch.rand(4,forward.dimension(model.case),device=device,dtype=torch.float64).requires_grad_()
    poly=Polynomial();u=poly(probe);a=poly.coefficients
    if model.case=='Heat2D':expected=2*probe[:,2:3]-2*a[0]-2*a[1]
    elif model.case=='AllenCahn':expected=2*probe[:,1:2]-2*a[0]+a[1]*(u**3-u)
    elif model.case=='Wave1D':expected=torch.ones_like(u)*(2-2*a[0])
    else:expected=(-4-a[0]*u-poly.forcing(probe))/poly.residual_scale
    analytic_residual_error=float((residual(poly,probe)-expected).abs().max().detach())
    assert analytic_residual_error<1e-9
    with torch.no_grad():
        for p in model.network.parameters():p.add_(.003*torch.randn_like(p))
    z=torch.rand(5,forward.dimension(model.case),device=device,dtype=torch.float64).requires_grad_()
    scale=torch.linspace(.4,.9,z.shape[1],device=device,dtype=torch.float64)
    q=((z.detach()-.17)/scale).requires_grad_();errors=[]
    for axis in range(z.shape[1]):
        a,b=model(z),model(q*scale+.17)
        for _ in range(3):
            a=forward.grad(a,z)[:,axis:axis+1];b=forward.grad(b,q)[:,axis:axis+1]/scale[axis]
            assert torch.allclose(a,b,rtol=1e-7,atol=1e-8)
            errors.append(float((a-b).abs().max().detach()))
    objective=residual(model,z).square().mean()
    analytic=torch.autograd.grad(objective,model.log_coefficients)[0]
    initial=model.log_coefficients.detach().clone();numeric=[]
    for i in range(len(initial)):
        vals=[]
        for sign in (1.,-1.):
            with torch.no_grad():model.log_coefficients.copy_(initial);model.log_coefficients[i].add_(sign*1e-6)
            vals.append(float(residual(model,z).square().mean().detach()))
        numeric.append((vals[0]-vals[1])/2e-6)
    with torch.no_grad():model.log_coefficients.copy_(initial)
    assert torch.allclose(analytic,torch.tensor(numeric,device=device,dtype=torch.float64),rtol=2e-4,atol=1e-8)
    assert torch.all(analytic.abs()>1e-12)
    residual(model,z).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    restored=Model(spec).to(device).double();restored.load_state_dict(model.state_dict());assert torch.equal(model(z),restored(z))
    zd,yd,s=load_observations(spec,device);assert len(zd)==800 and s>0
    if model.case=='Helmholtz2D':
        old=model.forcing(z).detach().clone()
        with torch.no_grad():model.log_coefficients.add_(.5)
        assert torch.equal(model.forcing(z).detach(),old)
    return dict(passed=True,spec=spec,device=device,optimizer_steps=0,parameter_finite_difference=True,
        physical_derivative_max_difference=max(errors),finite_backward=True,reload_exact=True,
        forcing_independent_of_estimate=True,analytic_residual_error=analytic_residual_error,hashes=hashes(spec))

def evaluate(model,folder,smoke=False):
    spec=model.spec;path=forward.DATA/(prefix(spec)+'.npz')
    with np.load(path) as data:z,y=data['coordinates'],data['target']
    with np.load(DATA/(prefix(spec)+'_split.npz')) as split:test=split['test']
    indices=np.arange(len(z))[::128 if smoke else 1];mask=np.isin(indices,test)
    predictions=[]
    with torch.no_grad():
        for start in range(0,len(indices),2048):
            predictions.append(model(torch.tensor(z[indices[start:start+2048]],device='cuda',dtype=torch.float64)).cpu().numpy())
    pred=np.concatenate(predictions);truth=y[indices]
    rng=torch.Generator(device='cuda').manual_seed(20261018)
    zz=forward.sample(spec,64 if smoke else 1024,rng,'cuda').double().detach().requires_grad_()
    pde=float(residual(model,zz).square().mean().sqrt().detach())
    coeff=model.coefficients.detach().cpu().numpy();reference=reference_coefficients(spec)
    relative=lambda a,b:float(100*np.linalg.norm(a-b)/np.linalg.norm(b))
    np.savez_compressed(folder/'prediction.npz',coordinates=z[indices],reference=truth,prediction=pred,heldout=mask,indices=indices)
    return dict(heldout_L2_percent=relative(pred[mask],truth[mask]),field_L2_percent=relative(pred,truth),
        coefficients=coeff.tolist(),reference_coefficients=reference.tolist(),coefficient_names=NAMES[spec['case']],
        coefficient_errors_percent=(100*np.abs(coeff/reference-1)).tolist(),
        derived_physical_parameter=float(np.sqrt(coeff[0])) if spec['case'] in ('Helmholtz2D','Wave1D') else None,
        independent_physics_rms=pde,reference_sha256=forward.sha(path),heldout_points=int(mask.sum()))

def train(spec,folder,smoke=False):
    assert platform.system()=='Linux' and torch.cuda.is_available(), 'Cloud only'
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'final.pt').exists() and not (folder/'history.json').exists(), 'No implicit retry'
    torch.set_num_threads(1);torch.manual_seed(spec['seed']);np.random.seed(spec['seed']);random.seed(spec['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=config(spec,smoke);evidence=hashes(spec);model=Model(spec).cuda();zd,yd,scale=load_observations(spec,'cuda')
    rng=torch.Generator(device='cuda').manual_seed(spec['seed']+1000)
    drng=torch.Generator(device='cuda').manual_seed(spec['seed']+2000)
    snapshots=folder/'snapshots';snapshots.mkdir(exist_ok=True);history=[];start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    def snapshot(stage,step,opt=None,scheduler=None):
        path=snapshots/(stage+'_'+str(step)+'.pt');assert not path.exists()
        value=dict(model=model.state_dict(),optimizer=opt.state_dict() if opt else None,
            scheduler=scheduler.state_dict() if scheduler else None,stage=stage,step=step,spec=spec,
            config=cfg,hashes=evidence,sampling_rng=rng.get_state(),data_rng=drng.get_state(),
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        temp=path.with_suffix('.tmp');torch.save(value,temp);os.replace(temp,path)
    def record(stage,step,total,data,physics,steps):
        row=dict(stage=stage,step=step,stage_steps=steps,adam_steps=cfg['adam_steps'],loss=float(total.detach()),
            data=float(data.detach()),physics=float(physics.detach()),coefficients=model.coefficients.detach().cpu().tolist(),
            coefficient_names=NAMES[model.case],elapsed_seconds=time.perf_counter()-start)
        history.append(row);forward.save(folder/'history.json',history)
        forward.save(folder/'progress.json',dict(status='training',**row));print(json.dumps(row),flush=True)
    def parts(z,ids):
        data=(model(zd[ids])-yd[ids]).square().mean()/scale;pde=residual(model,z).square().mean()
        return data+pde,data,pde
    snapshot('initial',0)
    model.log_coefficients.requires_grad_(False)
    opt=torch.optim.Adam(model.network.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,max(1,cfg['pretrain_steps']//3),gamma=.5)
    for step in range(1,cfg['pretrain_steps']+1):
        ids=torch.randint(len(zd),(cfg['batch'],),device='cuda',generator=drng)
        opt.zero_grad(set_to_none=True);data=(model(zd[ids])-yd[ids]).square().mean()/scale
        assert torch.isfinite(data);data.backward();torch.nn.utils.clip_grad_norm_(model.network.parameters(),1.,error_if_nonfinite=True)
        opt.step();scheduler.step()
        if step==1 or step%100==0 or step==cfg['pretrain_steps']:record('Pretrain',step,data,data,data.detach()*0,cfg['pretrain_steps'])
        if step%500==0 or step==cfg['pretrain_steps']:snapshot('Pretrain',step,opt,scheduler)
    model.log_coefficients.requires_grad_(True)
    opt=torch.optim.Adam([dict(params=model.network.parameters(),lr=cfg['network_lr']),dict(params=[model.log_coefficients],lr=cfg['coefficient_lr'])])
    scheduler=torch.optim.lr_scheduler.StepLR(opt,max(1,cfg['adam_steps']//2),gamma=.5)
    for step in range(1,cfg['adam_steps']+1):
        z=forward.sample(spec,cfg['nf'],rng,'cuda');ids=torch.randint(len(zd),(cfg['batch'],),device='cuda',generator=drng)
        opt.zero_grad(set_to_none=True);total,data,pde=parts(z,ids);assert torch.isfinite(total)
        total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();scheduler.step()
        if step==1 or step%100==0 or step==cfg['adam_steps']:record('Adam',step,total,data,pde,cfg['adam_steps'])
        if step%500==0 or step==cfg['adam_steps']:snapshot('Adam',step,opt,scheduler)
    z=forward.sample(spec,cfg['lbfgs_nf'],rng,'cuda');ids=torch.arange(len(zd),device='cuda')
    opt=torch.optim.LBFGS(model.parameters(),lr=.8,max_iter=cfg['lbfgs_max_iter'],max_eval=2*cfg['lbfgs_max_iter'],
        history_size=50,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-10)
    calls=0
    def closure():
        nonlocal calls
        opt.zero_grad(set_to_none=True);total,data,pde=parts(z,ids);assert torch.isfinite(total);total.backward();calls+=1
        if calls==1 or calls%25==0:record('L-BFGS',calls,total,data,pde,cfg['lbfgs_max_iter'])
        return total
    assert float(closure().detach())==float(closure().detach())
    opt.step(closure);snapshot('LBFGS_accepted_final',int(next(iter(opt.state.values())).get('n_iter',0)),opt)
    torch.save(model.state_dict(),folder/'final.pt');torch.cuda.synchronize()
    seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()
    model.load_state_dict(torch.load(folder/'final.pt',weights_only=True,map_location='cuda'));model.double()
    result=dict(spec=spec,config=cfg,hashes=evidence,weights_sha256=forward.sha(folder/'final.pt'),
        metrics=evaluate(model,folder,smoke),parameters=sum(p.numel() for p in model.parameters()),
        training_seconds=seconds,peak_cuda_memory_bytes=peak,closure_calls=calls,
        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,internal_label_count=len(zd))
    forward.save(folder/'result.json',result)
    forward.save(folder/'snapshot_manifest.json',dict(files={str(p.relative_to(folder)):forward.sha(p) for p in snapshots.glob('*.pt')}))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-data',action='store_true');parser.add_argument('--task')
    parser.add_argument('--output',type=Path);parser.add_argument('--device',default='cpu');args=parser.parse_args()
    if args.prepare_data:print(json.dumps(prepare_data()))
    else:
        spec=next(s for s in jobs() if s['id']==args.task)
        forward.save(args.output,self_test(spec,args.device))
