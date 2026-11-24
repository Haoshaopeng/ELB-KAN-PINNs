"""Observed-data inverse PDE study using the frozen variable-basis architecture."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn

import adaptive_basis as study
from forward_training import FROZEN_HASHES

HERE = Path(__file__).resolve().parent
ROOT = HERE/'adaptive_input_basis'/'labeled_inverse_experiments'
DATA = HERE/'adaptive_input_basis'/'fixed_inverse_observations'
base = study.base
INITIAL = {'KdV':[.005], 'NS_periodic':[1.5, .02]}


def digest():
    frozen = study.digest()
    assert frozen == FROZEN_HASHES, 'Frozen forward model implementation changed'
    for name in ['forward_training.py', Path(__file__).name]:
        frozen[name] = base.sha(HERE/name)
    return frozen


def config(case, smoke=False):
    return dict(pretrain_steps=3 if smoke else 3000,
        adam_steps=6 if smoke else 8000 if case == 'KdV' else 6000,
        nf=16 if smoke else 1024 if case == 'KdV' else 384,
        nb=8 if smoke else 128 if case == 'KdV' else 64,
        batch=32 if smoke else 800 if case == 'KdV' else 1024,
        lbfgs_max_iter=3 if smoke else 400, precision='FP32',
        initial_unknown_coefficients=INITIAL[case],
        unknown_parameters=['lambda2'] if case == 'KdV' else ['lambda1','lambda2'],
        known_lambda1=1. if case == 'KdV' else None,
        noise=0, loss='normalized observation MSE + PDE MSE + 10 periodic BC',
        pretrain_loss='normalized observation MSE; PDE coefficients frozen',
        network_lr=5e-4, coefficient_lr=2e-3, gradient_clip=1.,
        hard_initial_condition=True, truth_penalty=False, checkpoint='final only', smoke=smoke)


class InverseModel(nn.Module):
    def __init__(self, case, variant):
        super().__init__()
        self.case, self.variant = case, variant
        self.network = study.AdaptiveModel(case, variant)
        # Remove the forward-only fixed coefficients. Residuals use the property below.
        del self.network.coefficients
        self.log_coefficients = nn.Parameter(torch.tensor([math.log(v) for v in INITIAL[case]]))

    @property
    def coefficients(self):
        values = self.log_coefficients.exp()
        return torch.cat([torch.ones_like(values), values]) if self.case == 'KdV' else values

    @property
    def kan(self):
        return self.network.kan

    def correction(self, z):
        return self.network.correction(z)

    def forward(self, z):
        return self.network(z)


def residual(model, z):
    if model.case == 'KdV':
        u = model(z)
        d = base.derivative(u, z)
        uxx = base.derivative(d[:, :1], z)[:, :1]
        uxxx = base.derivative(uxx, z)[:, :1]
        return d[:, 1:2] + u*d[:, :1] + model.coefficients[1]*uxxx
    fu, fv, _ = base.ns.momentum_residual(model, z)
    return torch.cat([fu, fv], 1)


def prediction(model, z):
    if model.case == 'KdV':
        return model(z)
    u, v, _ = base.ns.velocity_fields(model, z)
    return torch.cat([u, v], 1)


def observation_loss(model, z, target, scale):
    z = z.detach().requires_grad_()
    return (prediction(model, z)-target).square().mean()/scale


def parts(model, z, bc, zd, yd, scale):
    data = observation_loss(model, zd, yd, scale)
    physics = residual(model, z).square().mean()
    boundary = base.boundary(model, bc)
    return data+physics+10*boundary, data, physics, boundary


def data_hashes(case):
    return {case+suffix:base.sha(DATA/(case+suffix)) for suffix in ('_observations.npz','_split.npz')}


def load_observations(case, device):
    with np.load(DATA/(case+'_observations.npz')) as data, np.load(DATA/(case+'_split.npz')) as split:
        assert not np.intersect1d(data['indices'], split['test']).size
        assert np.array_equal(data['indices'], split['train'])
        z = torch.tensor(data['coordinates'], dtype=torch.float32, device=device)
        y = torch.tensor(data['target'], dtype=torch.float32, device=device)
    scale = y.square().mean().detach()
    assert torch.isfinite(scale) and scale > 0
    return z, y, scale


def self_test(device):
    reports = []
    for case in ('KdV','NS_periodic'):
        zd, yd, scale = load_observations(case, device)
        assert len(zd) == (800 if case == 'KdV' else 10000)
        for variant in study.VARIANTS:
            base.seed(42)
            model = InverseModel(case, variant).to(device).double()
            assert torch.allclose(model.log_coefficients.exp(), torch.tensor(INITIAL[case], device=device, dtype=torch.float64))
            with torch.no_grad():
                for p in model.network.parameters():
                    p.add_(.003*torch.randn_like(p))
            # Non-unit affine scales independently verify physical derivatives.
            dim = 2 if case == 'KdV' else 3
            q = torch.rand(5, dim, device=device, dtype=torch.float64).requires_grad_()
            scales = torch.tensor([.7,.3] if dim == 2 else [.7,1.3,.3], device=device, dtype=torch.float64)
            z = (q.detach()*scales+.17).requires_grad_()
            p = model(q*scales+.17)
            def grad(v):
                return base.derivative(v, q)/scales
            if case == 'KdV':
                du = grad(p)
                manual = du[:,1:2]+p*du[:,:1]+model.coefficients[1]*grad(grad(du[:,:1])[:,:1])[:,:1]
            else:
                ds, dp = grad(p[:,:1]), grad(p[:,1:2])
                u, v = ds[:,1:2], -ds[:,:1]
                du, dv = grad(u), grad(v)
                lu = grad(du[:,:1])[:,:1]+grad(du[:,1:2])[:,1:2]
                lv = grad(dv[:,:1])[:,:1]+grad(dv[:,1:2])[:,1:2]
                l1,l2 = model.coefficients
                manual = torch.cat([du[:,2:3]+l1*(u*du[:,:1]+v*du[:,1:2])+dp[:,:1]-l2*lu,
                                    dv[:,2:3]+l1*(u*dv[:,:1]+v*dv[:,1:2])+dp[:,1:2]-l2*lv],1)
            chain = float((manual-residual(model,z)).abs().max().detach())
            assert chain < 1e-9, (case,variant,chain)
            objective = residual(model,z).square().mean()
            analytical = torch.autograd.grad(objective, model.log_coefficients)[0]
            numeric = []
            for i in range(len(model.log_coefficients)):
                initial = model.log_coefficients.detach().clone()
                values = []
                for sign in (1.,-1.):
                    with torch.no_grad():
                        model.log_coefficients.copy_(initial)
                        model.log_coefficients[i].add_(sign*1e-6)
                    values.append(float(residual(model,z).square().mean().detach()))
                with torch.no_grad():
                    model.log_coefficients.copy_(initial)
                numeric.append((values[0]-values[1])/2e-6)
            numeric = torch.tensor(numeric, device=device, dtype=torch.float64)
            assert torch.allclose(analytical,numeric,rtol=2e-4,atol=1e-8)
            assert (analytical.abs()>1e-12).all(), 'Coefficient gradient is inactive'
            bc = torch.rand(4,dim-1,device=device,dtype=torch.float64)
            if dim == 3:
                bc[:,0] *= 2*math.pi
            total,_,_,_ = parts(model,z,bc,zd[:8].double(),yd[:8].double(),scale.double())
            total.backward()
            assert torch.isfinite(total)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            if variant == 'Adaptive':
                assert model.kan[0].logits.grad.abs().max()>1e-14
            zi=z.detach().clone()
            zi[:,-1]=0.
            zi.requires_grad_()
            if case == 'KdV':
                ic=float((model(zi)-torch.cos(math.pi*zi[:,:1])).abs().max().detach())
            else:
                x,y=zi[:,:1],zi[:,1:2]
                target=torch.cat([.3+.5*x.sin()*y.cos()+.15*(2*x).cos()*y.cos()-.3*x.sin()*(3*y).sin(),
                                  .2-.5*x.cos()*y.sin()+.3*(2*x).sin()*y.sin()-.1*x.cos()*(3*y).cos()],1)
                ic=float((prediction(model,zi)-target).abs().max().detach())
            assert ic<1e-12
            reports.append(dict(case=case,variant=variant,chain_error=chain,hard_initial_error=ic,
                coefficient_gradient=analytical.detach().cpu().tolist(),coefficient_finite_difference_passed=True,
                finite_backward=True,parameters=sum(p.numel() for p in model.parameters())))
    return dict(passed=True, device=device, models=reports, optimizer_steps=0, source_hashes=digest(),
                datasets={case:data_hashes(case) for case in ('KdV','NS_periodic')})


def evaluate(model, folder, smoke):
    model.double()
    case = model.case
    if case == 'KdV':
        path = HERE/'benchmark_audit/KdV_crosschecked_reference.mat'
        ref = base.loadmat(path)
        x,t = ref['x'].ravel(),ref['t'].ravel()
        xx,tt = np.meshgrid(x,t,indexing='ij')
        z = np.stack([xx.ravel(),tt.ravel()],1)
        target = ref['usol'].ravel()[:,None]
        truth = np.array([1.,.0025])
    else:
        path = HERE/'NSdiagnostics/independent_solution_diagnostic_data.npz'
        with np.load(path) as ref:
            z,target,truth = ref['coordinates'],ref['velocity'],ref['truth']
    with np.load(DATA/(case+'_split.npz')) as split:
        test = split['test']
    selected = np.flatnonzero(z[:,-1]>0)
    if smoke:
        selected = selected[::128]
    preds = []
    for start in range(0,len(selected),2048):
        batch=torch.tensor(z[selected[start:start+2048]],device='cuda',dtype=torch.float64,requires_grad=True)
        preds.append(prediction(model,batch).detach().cpu().numpy())
    pred = np.concatenate(preds)
    y = target[selected]
    heldout = np.isin(selected,test)
    assert heldout.any()
    relative = lambda a,b:float(100*np.linalg.norm(a-b)/np.linalg.norm(b))
    coefficients = model.coefficients.detach().cpu().numpy()
    rng=torch.Generator(device='cuda').manual_seed(20261011)
    zz,bb=base.sample(case,64 if smoke else 1024,32 if smoke else 256,rng)
    zz,bb=zz.double().detach().requires_grad_(),bb.double()
    physics=float(residual(model,zz).square().mean().sqrt().detach())
    boundary=float(base.boundary(model,bb).detach())
    np.savez_compressed(folder/'final_prediction.npz',indices=selected,coordinates=z[selected],reference=y,
                        prediction=pred,heldout=heldout)
    gates=None
    if model.variant in ('Adaptive','Uniform'):
        gates=dict(variables=['x','t'] if case=='KdV' else ['x','y','t'],families=study.FAMILIES,
                   weights=model.kan[0].weights().detach().cpu().tolist(),
                   warning='Soft mixtures; weights do not identify causal basis importance')
    return dict(field_L2_percent_excluding_initial_time=relative(pred,y),
        heldout_L2_percent=relative(pred[heldout],y[heldout]),heldout_points=int(heldout.sum()),
        primary_metric='heldout_L2_percent; observation points excluded',
        coefficients=coefficients.tolist(),reference_coefficients=truth.tolist(),
        lambda1_error_percent=float(abs(coefficients[0]/truth[0]-1)*100),
        lambda2_error_percent=float(abs(coefficients[1]/truth[1]-1)*100),
        lambda1_estimated=case!='KdV',independent_physics_rms=physics,
        independent_boundary_loss=boundary,reference_sha256=base.sha(path),gate_diagnostic=gates)


def train(job,root,smoke=False):
    cfg=config(job['case'],smoke)
    folder=root/(job['case']+'_'+job['variant']+'_seed'+str(job['seed']))
    folder.mkdir(parents=True,exist_ok=True)
    hashes,data_provenance=digest(),data_hashes(job['case'])
    if (folder/'result.json').exists():
        result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        assert result['config']==cfg and result['source_hashes']==hashes and result['data_hashes']==data_provenance
        assert base.sha(folder/'final.pt')==result['weights_sha256']
        return result
    assert not (folder/'checkpoint.pt').exists(), 'Partial attempt preserved; explicit resume is required'
    base.seed(job['seed'])
    model=InverseModel(job['case'],job['variant']).cuda()
    zd,yd,scale=load_observations(job['case'],'cuda')
    rng=torch.Generator(device='cuda').manual_seed(job['seed']+1000)
    drng=torch.Generator(device='cuda').manual_seed(job['seed']+2000)
    history=[]
    start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def record(stage,step,values,stage_steps):
        total,data,physics,boundary=values
        row=dict(job=job,stage=stage,step=step,stage_steps=stage_steps,adam_steps=cfg['adam_steps'],
            loss=float(total.detach()),data=float(data.detach()),physics=float(physics.detach()),
            boundary=float(boundary.detach()),coefficients=model.coefficients.detach().cpu().tolist(),
            elapsed_seconds=time.perf_counter()-start)
        if job['variant'] in ('Adaptive','Uniform'):
            row['gates']=model.kan[0].weights().detach().cpu().tolist()
        history.append(row)
        base.save(root/'progress.json',dict(status='training',**row))
        base.save(folder/'history.json',history)
        print(json.dumps(row),flush=True)

    def save_checkpoint(stage,step,opt,scheduler=None):
        study.checkpoint(folder/'checkpoint.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),
            scheduler=scheduler.state_dict() if scheduler else None,sampling_rng=rng.get_state(),
            data_rng=drng.get_state(),source_hashes=hashes,data_hashes=data_provenance,
            stage=stage,step=step,config=cfg,job=job))

    model.log_coefficients.requires_grad_(False)
    opt=torch.optim.Adam(model.network.parameters(),lr=1e-3)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,max(1,cfg['pretrain_steps']//3),gamma=.5)
    for step in range(1,cfg['pretrain_steps']+1):
        idx=torch.randint(len(zd),(cfg['batch'],),device='cuda',generator=drng)
        opt.zero_grad(set_to_none=True)
        data=observation_loss(model,zd[idx],yd[idx],scale)
        assert torch.isfinite(data)
        data.backward()
        torch.nn.utils.clip_grad_norm_(model.network.parameters(),1.,error_if_nonfinite=True)
        opt.step(); scheduler.step()
        if step==1 or step%100==0 or step==cfg['pretrain_steps']:
            record('Pretrain',step,(data,data,data.detach()*0,data.detach()*0),cfg['pretrain_steps'])
        if step%500==0 or step==cfg['pretrain_steps']:
            save_checkpoint('Pretrain',step,opt,scheduler)
    model.log_coefficients.requires_grad_(True)
    opt=torch.optim.Adam([{'params':model.network.parameters(),'lr':cfg['network_lr']},
                          {'params':[model.log_coefficients],'lr':cfg['coefficient_lr']}])
    scheduler=torch.optim.lr_scheduler.StepLR(opt,max(1,cfg['adam_steps']//2),gamma=.5)
    for step in range(1,cfg['adam_steps']+1):
        z,bc=base.sample(job['case'],cfg['nf'],cfg['nb'],rng)
        idx=torch.randint(len(zd),(cfg['batch'],),device='cuda',generator=drng)
        opt.zero_grad(set_to_none=True)
        values=parts(model,z,bc,zd[idx],yd[idx],scale)
        assert torch.isfinite(values[0])
        values[0].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step(); scheduler.step()
        if step==1 or step%100==0 or step==cfg['adam_steps']:
            record('Adam',step,values,cfg['adam_steps'])
        if step%500==0 or step==cfg['adam_steps']:
            save_checkpoint('Adam',step,opt,scheduler)
    z,bc=base.sample(job['case'],2*cfg['nf'],2*cfg['nb'],rng)
    idx=torch.randperm(len(zd),device='cuda',generator=drng)[:32 if smoke else 2048]
    opt=torch.optim.LBFGS(model.parameters(),lr=.8,max_iter=cfg['lbfgs_max_iter'],max_eval=2*cfg['lbfgs_max_iter'],
        history_size=50,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-10)
    calls=0
    def closure():
        nonlocal calls
        opt.zero_grad(set_to_none=True)
        values=parts(model,z,bc,zd[idx],yd[idx],scale)
        assert torch.isfinite(values[0])
        values[0].backward()
        calls+=1
        if calls==1 or calls%25==0:
            record('L-BFGS',calls,values,cfg['lbfgs_max_iter'])
        return values[0]
    assert float(closure().detach())==float(closure().detach())
    opt.step(closure)
    study.checkpoint(folder/'final.pt',model.state_dict())
    torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    peak=torch.cuda.max_memory_allocated()
    model.load_state_dict(torch.load(folder/'final.pt',map_location='cuda',weights_only=True))
    # Only now open the full reference; training used the observation-only files.
    metrics=evaluate(model,folder,smoke)
    result=dict(job=job,config=cfg,source_hashes=hashes,data_hashes=data_provenance,
        weights_sha256=base.sha(folder/'final.pt'),parameters=sum(p.numel() for p in model.parameters()),
        training_seconds=seconds,peak_cuda_memory_bytes=peak,closure_calls=calls,
        lbfgs_iterations=int(next(iter(opt.state.values())).get('n_iter',0)),
        observation_count=len(zd),internal_label_count=int((zd[:,-1]>0).sum().item()),
        exploratory_not_manuscript_result=True,gpu=torch.cuda.get_device_name(0),
        python=platform.python_version(),torch=torch.__version__,**metrics)
    base.save(folder/'result.json',result)
    base.save(folder/'files_sha256.json',{p.name:base.sha(p) for p in folder.iterdir()
              if p.is_file() and p.name!='files_sha256.json'})
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--node',choices=['A','B','C'],required=True)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--cpu-self-test',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.cpu_self_test:
        assert args.self_test and not args.smoke
        report=self_test('cpu')
        base.save(ROOT/'cpu_self_test.json',report)
        print('CPU_ONLY_SELF_TEST_PASSED; optimizer_steps=0',flush=True)
        return
    assert platform.system()=='Linux', 'Cloud-only training'
    assert torch.cuda.is_available() and torch.__version__=='2.5.1+cu121'
    root=ROOT/('smoke_'+args.node if args.smoke else args.node)
    root.mkdir(parents=True,exist_ok=True)
    if args.self_test:
        base.save(root/'self_test.json',self_test('cuda'))
        print('INVERSE_SELF_TEST_PASSED',flush=True)
        return
    test=json.loads((ROOT/args.node/'self_test.json').read_text(encoding='utf-8'))
    assert test['passed'] and test['source_hashes']==digest()
    assert test['datasets']=={case:data_hashes(case) for case in ('KdV','NS_periodic')}
    seed={'A':42,'B':43,'C':44}[args.node]
    jobs=[dict(case=c,variant=v,seed=seed) for c in ('KdV','NS_periodic') for v in study.VARIANTS]
    if args.smoke:
        jobs=[dict(case=c,variant='Adaptive',seed=seed) for c in ('KdV','NS_periodic')]
    manifest=dict(node=args.node,jobs=jobs,source_hashes=digest(),
        budgets={c:config(c,args.smoke) for c in ('KdV','NS_periodic')},
        datasets={c:data_hashes(c) for c in ('KdV','NS_periodic')},
        purpose='Observed-data inverse companion, not a label-only ablation of the forward study',
        all_seeds_reported=True,truth_prior=False,
        limitations=['Known initial conditions remain hard constraints',
            'Different problem from fixed-coefficient forward solving',
            'Soft input-variable mixtures, not spatial partitions or identifiable importance',
            'Three seeds; all runs retained; no manuscript changes without review'])
    import fcntl
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'manifest.json'
        if path.exists():
            assert json.loads(path.read_text(encoding='utf-8'))==manifest
        else:
            base.save(path,manifest)
        results=[]
        for job in jobs:
            try:
                result=train(job,root,args.smoke)
            except Exception as exc:
                base.save(root/'progress.json',dict(status='failed',job=job,error=repr(exc)))
                raise
            results.append(result)
            base.save(root/'completed.json',results)
            torch.cuda.empty_cache()
        base.save(root/'progress.json',dict(status='complete',completed=len(results),jobs=len(jobs)))


if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    main()
