"""Paired-seed supplementary experiments; audited references, full budgets.

No existing experiments or manuscript assets are overwritten. Test errors are
never used for checkpoint selection. Smoke results live in a separate tree.
"""
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.io import loadmat
from scipy.fft import dct
import torch
from torch import nn

CODE = Path(__file__).resolve().parent
OUT = CODE / 'auxiliary_suite'
SEEDS = [42, 43, 44, 45, 46]
CASES = ['Burgers', 'KdV', 'NS_wake', 'NS_periodic']
PRIMARY = {'Burgers': 'Chebyshev', 'KdV': 'Fourier', 'NS_wake': 'Chebyshev', 'NS_periodic': 'Chebyshev'}
NS_NAMES = {'NS_wake': 'cylinder_wake', 'NS_periodic': 'independent_solution'}


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def modules():
    return (load('supp_burgers', 'burgers_hybrid.py'),
            load('supp_kdv', 'kdv_inverse_model.py'),
            load('supp_ns', 'navier_stokes_diagnostics.py'))


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)


class BasisLayer(nn.Module):
    def __init__(self, inp, out, terms, basis, scale, squash=False):
        super().__init__()
        assert terms % 2 == 0
        self.coeff = nn.Parameter(torch.randn(inp, out, terms) * scale)
        self.terms, self.basis, self.squash = terms, basis, squash

    def forward(self, x):
        x = torch.tanh(x) if self.squash else x
        if self.basis == 'Chebyshev':
            values = [torch.ones_like(x), x]
            for _ in range(2, self.terms):
                values.append(2*x*values[-1]-values[-2])
            values = torch.stack(values, -1)
        else:
            freq = torch.arange(1, self.terms//2+1, device=x.device, dtype=x.dtype)*math.pi
            phase = x.unsqueeze(-1)*freq
            values = torch.cat([phase.cos(), phase.sin()], -1)
        return torch.einsum('nik,iok->no', values, self.coeff)


def mlp(inp, width, depth, out, activation='tanh', xavier=False):
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(inp, width), nn.Tanh() if activation == 'tanh' else Sine()]
        inp = width
    layers.append(nn.Linear(width, out))
    result = nn.Sequential(*layers)
    if xavier:
        for layer in result:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    return result


class Sine(nn.Module):
    def forward(self, x):
        return x.sin()


class ScalarModel(nn.Module):
    def __init__(self, case, variant):
        super().__init__()
        self.case, self.variant = case, variant
        self.has_mlp = variant != 'KAN_only'
        self.has_kan = variant not in ('MLP', 'MLP_matched')
        basis = PRIMARY[case] if variant in ('Hybrid', 'KAN_only') else variant
        if case == 'Burgers':
            width = 104 if variant == 'MLP' else 103 if variant == 'MLP_matched' else 64
            depth, kw, terms, activation, scale = 3, 32, 12, 'tanh', .1
        else:
            width = 64 if variant == 'MLP_matched' else 40
            depth = 6 if variant in ('MLP', 'MLP_matched') else 3
            kw, terms, activation, scale = 32, 16, 'sine', .01
            self.log_lambda2 = nn.Parameter(torch.log(torch.tensor([.005])))
            self.lambda1 = 1.
        if self.has_mlp:
            self.mlp = mlp(2, width, depth, 1, activation, case == 'KdV')
        if self.has_kan:
            self.kan = nn.Sequential(BasisLayer(2, kw, terms, basis, scale, case == 'KdV'), nn.Tanh(),
                                     BasisLayer(kw, kw, terms, basis, scale, case == 'KdV'), nn.Tanh(), nn.Linear(kw, 1))
            if self.has_mlp:
                nn.init.zeros_(self.kan[-1].weight)
                nn.init.zeros_(self.kan[-1].bias)

    @property
    def lambda2(self):
        return self.log_lambda2.exp()

    def branch_outputs(self, x, t):
        z = torch.cat([x, 2*t-1 if self.case == 'Burgers' else t], 1)
        zero = x*0
        g = self.mlp(z) if self.has_mlp else zero
        r = self.kan(z) if self.has_kan else zero
        return (t*g, t*r) if self.case == 'KdV' else (g, r)

    def forward(self, x, t):
        g, r = self.branch_outputs(x, t)
        return g+r+(torch.cos(math.pi*x) if self.case == 'KdV' else 0)


class NSModel(nn.Module):
    def __init__(self, bounds, variant, init):
        super().__init__()
        self.register_buffer('lower', torch.tensor(bounds[0], dtype=torch.float32))
        self.register_buffer('upper', torch.tensor(bounds[1], dtype=torch.float32))
        self.has_mlp = variant != 'KAN_only'
        self.has_kan = variant not in ('MLP', 'MLP_matched')
        if self.has_mlp:
            self.mlp = mlp(3, 82 if variant == 'MLP_matched' else 64, 3, 2, xavier=True)
            if variant == 'MLP_matched':
                self.mlp[4] = nn.Linear(82,81)
                self.mlp[6] = nn.Linear(81,2)
                for layer in [self.mlp[4],self.mlp[6]]:
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
        if self.has_kan:
            basis = 'Fourier' if variant == 'Fourier' else 'Chebyshev'
            self.kan = nn.Sequential(BasisLayer(3, 24, 8, basis, .02), nn.Tanh(),
                                     BasisLayer(24, 24, 8, basis, .02), nn.Tanh(), nn.Linear(24, 2))
            if self.has_mlp:
                nn.init.zeros_(self.kan[-1].weight)
                nn.init.zeros_(self.kan[-1].bias)
        self.log_lambda = nn.Parameter(torch.tensor(np.log(init), dtype=torch.float32))

    def branch_outputs(self, z):
        q = 2*(z-self.lower)/(self.upper-self.lower)-1
        return (self.mlp(q) if self.has_mlp else q[:, :2]*0,
                self.kan(q) if self.has_kan else q[:, :2]*0)

    def forward(self, z):
        g, r = self.branch_outputs(z)
        return g+r

    @property
    def coefficients(self):
        return self.log_lambda.exp()


def plan():
    jobs = []
    def add(case, variant, seed, noise=0., init=(1.5, .02)):
        if case == 'KdV' and init == (1.5,.02):
            init = (1.,.005)
        jobs.append(dict(case=case, variant=variant, seed=seed, noise=noise, init=list(init)))
    for case in CASES:
        for seed in SEEDS:
            for variant in ['MLP', 'Hybrid', 'MLP_matched']:
                add(case, variant, seed)
        other = 'Chebyshev' if PRIMARY[case] == 'Fourier' else 'Fourier'
        for seed in SEEDS[:3]:
            for variant in [other, 'KAN_only']:
                add(case, variant, seed)
    for case in ['KdV', 'NS_wake', 'NS_periodic']:
        for seed in SEEDS[:3]:
            for variant in ['MLP', 'Hybrid']:
                add(case, variant, seed, .05)
    for case in ['NS_wake', 'NS_periodic']:
        for seed in SEEDS[:3]:
            for variant in ['MLP', 'Hybrid']:
                add(case, variant, seed, init=(.5, .005))
    for seed in SEEDS[:3]:
        for variant in ['MLP','Hybrid']:
            add('KdV',variant,seed,init=(1.,.001))
    return jobs


def job_id(job):
    return f"{job['case']}_{job['variant']}_seed{job['seed']}_noise{job['noise']:g}_initial{job['init'][0]:g}_{job['init'][1]:g}"


def evaluate_scalar(model, x, t, reference, heldout):
    model.double()
    X, T = np.meshgrid(x, t, indexing='ij')
    pred, g, r = [], [], []
    with torch.no_grad():
        for i in range(0, X.size, 4096):
            a = torch.tensor(X.ravel()[i:i+4096, None], device='cuda', dtype=torch.float64)
            b = torch.tensor(T.ravel()[i:i+4096, None], device='cuda', dtype=torch.float64)
            gg, rr = model.branch_outputs(a, b)
            pred.append(model(a,b).cpu().numpy().ravel())
            g.append(gg.cpu().numpy().ravel())
            r.append(rr.cpu().numpy().ravel())
    pred, g, r = (np.concatenate(v).reshape(reference.shape) for v in [pred,g,r])
    ratio = lambda a,b: float(100*np.linalg.norm(a-b)/np.linalg.norm(b))
    metrics = {'field_L2_percent':ratio(pred, reference), 'heldout_L2_percent':ratio(pred[heldout],reference[heldout])}
    if model.case == 'KdV':
        metrics.update(lambda2=float(model.lambda2.item()), lambda2_error_percent=abs(model.lambda2.item()/.0025-1)*100)
    return metrics, dict(x=x,t=t,reference=reference,prediction=pred,global_branch=g,residual_branch=r,heldout=heldout)


def train_scalar(job, folder, mods, smoke):
    burgers, kdv, _ = mods
    case = job['case']
    if case == 'Burgers':
        path = CODE/'Burgersaudit/Burgers_Cole_Hopfreference_on_same_grid.npz'
        a = np.load(path)
        x,t,u = a['x'],a['t'],a['usol']
        heldout = np.ones(u.shape, dtype=bool)
    else:
        path = CODE/'benchmark_audit/KdV_crosschecked_reference.mat'
        a = loadmat(path)
        x,t,u = a['x'].ravel(),a['t'].ravel(),a['usol']
        seed_all(42)
        xd,td,ud = kdv.sample_data(x,t,u,800)
        clean = ud.clone()
        gen = torch.Generator(device='cuda').manual_seed(20261010)
        ud = ud+(td>0)*job['noise']*clean.std()*torch.randn(ud.shape,device='cuda',generator=gen)
        observed = set(zip(xd.cpu().numpy().ravel().tolist(),td.cpu().numpy().ravel().tolist()))
        X,T = np.meshgrid(x,t,indexing='ij')
        heldout = np.array([(float(np.float32(xx)),float(np.float32(tt))) not in observed for xx,tt in zip(X.ravel(),T.ravel())]).reshape(u.shape)
        np.savez(folder/'training_labels.npz',x=xd.cpu().numpy(),t=td.cpu().numpy(),clean=clean.cpu().numpy(),noisy=ud.cpu().numpy(),heldout=heldout)
    seed_all(job['seed'])
    model = ScalarModel(case,job['variant']).cuda()
    if case == 'KdV':
        with torch.no_grad():
            model.log_lambda2.fill_(math.log(job['init'][1]))
    seed_all(job['seed']+1000)
    history, calls = [], 0
    def record(stage, step, values):
        history.append(dict(stage=stage,step=step,loss=float(values[0].detach()),parts=[float(v.detach()) for v in values[1:]]))
        if step % 1000 == 0 or step == 1:
            print(job_id(job),stage,step,history[-1]['loss'],flush=True)
    if case == 'Burgers':
        adam, lbmax, nf = (10,3,64) if smoke else (5000,1000,10000)
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
        best = float('inf')
        for i in range(adam):
            if i % 50 == 0:
                pts = burgers.sample_points(nf,400 if not smoke else 16,400 if not smoke else 16,.5,'cuda')
            optimizer.zero_grad(set_to_none=True)
            values = burgers.compute_loss(model,pts)
            assert torch.isfinite(values[0])
            values[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            # Save the state actually associated with the scored objective.
            if values[0].item() < best:
                best = values[0].item()
                torch.save(model.state_dict(),folder/'Adambest_training_loss_weights.pt')
            optimizer.step()
            record('adam',i+1,values)
        model.load_state_dict(torch.load(folder/'Adambest_training_loss_weights.pt',weights_only=True),strict=True)
        pts = burgers.sample_points(64 if smoke else 15000,16 if smoke else 500,16 if smoke else 500,.5,'cuda')
        losses = lambda: burgers.compute_loss(model,pts)
        config = dict(adam=adam,lbfgs_max_iter=lbmax,adam_nf=nf,lbfgs_nf=64 if smoke else 15000,focus=.5,resample_interval=50,loss='PDE + 10 IC + 10 BC',checkpoint='minimum training objective before Adam step, then final fixed-sample L-BFGS')
        options = dict(max_iter=lbmax,tolerance_grad=1e-7,tolerance_change=1e-9,line_search_fn='strong_wolfe')
    else:
        pre,joint,lbmax,nf = (10,10,3,64) if smoke else (8000,8000,400,2048)
        model.log_lambda2.requires_grad_(False)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(opt,max(1,pre//3),gamma=.5)
        for i in range(pre):
            opt.zero_grad(set_to_none=True)
            loss = (model(xd,td)-ud).square().mean()
            assert torch.isfinite(loss)
            loss.backward(); opt.step(); scheduler.step()
            record('pretrain',i+1,(loss,))
        model.log_lambda2.requires_grad_(True)
        opt = torch.optim.Adam(model.parameters(),lr=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(opt,max(1,joint//2),gamma=.5)
        def boundary(tb):
            left = -torch.ones_like(tb,requires_grad=True)
            right = torch.ones_like(tb,requires_grad=True)
            ul,ur = model(left,tb),model(right,tb)
            loss = (ul-ur).square().mean()
            for _ in (1,2):
                ul = torch.autograd.grad(ul,left,torch.ones_like(ul),create_graph=True)[0]
                ur = torch.autograd.grad(ur,right,torch.ones_like(ur),create_graph=True)[0]
                loss = loss+(ul-ur).square().mean()
            return loss
        def components(xf,tf,tb):
            ld = (model(xd,td)-ud).square().mean()
            lp = kdv.compute_pde_residual(model,xf,tf).square().mean()
            lb = boundary(tb)
            return 10*ld+.1*lp+lb,ld,lp,lb
        for i in range(joint):
            opt.zero_grad(set_to_none=True)
            values = components(2*torch.rand(nf,1,device='cuda')-1,torch.rand(nf,1,device='cuda'),torch.rand(128,1,device='cuda'))
            assert torch.isfinite(values[0])
            values[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step(); scheduler.step()
            record('physics',pre+i+1,values)
        xf=(2*torch.rand(nf,1,device='cuda')-1).requires_grad_()
        tf=torch.rand(nf,1,device='cuda',requires_grad=True)
        tb=torch.rand(128,1,device='cuda')
        losses=lambda:components(xf,tf,tb)
        config=dict(pretrain=pre,joint=joint,lbfgs_max_iter=lbmax,nf=nf,periodic_orders=[0,1,2],loss='10 data + 0.1 PDE + BC',checkpoint='final only')
        options=dict(max_iter=lbmax,history_size=100,line_search_fn='strong_wolfe',tolerance_change=1e-9)
    optimizer=torch.optim.LBFGS(model.parameters(),**options)
    def closure():
        nonlocal calls
        optimizer.zero_grad(set_to_none=True)
        values=losses()
        assert torch.isfinite(values[0])
        values[0].backward()
        calls+=1
        record('lbfgs',calls,values)
        return values[0]
    first=float(closure().detach()); second=float(closure().detach())
    assert first==second
    optimizer.step(closure)
    torch.save(model.state_dict(),folder/'final_weights.pt')
    metric,arrays=evaluate_scalar(model,x,t,u,heldout)
    save(folder/'training_history.json',history)
    return model,metric,arrays,dict(config,reference_sha256=sha(path),closure_calls=calls,fixed_closure_verified=True)


def train_ns(job,folder,mods,smoke):
    ns=mods[2]
    source=CODE/'NSdiagnostics'
    case=NS_NAMES[job['case']]
    path=source/f'{case}_diagnostic_data.npz'
    split_path=source/f'{case}_fixed_split.npz'
    data,split=np.load(path),np.load(split_path)
    train,test=split['train'],split['test']
    assert not np.intersect1d(train,test).size
    seed_all(job['seed'])
    model=NSModel(data['bounds'],job['variant'],job['init']).cuda()
    z_data=torch.tensor(data['coordinates'][train],device='cuda',dtype=torch.float32)
    clean=torch.tensor(data['velocity'][train],device='cuda',dtype=torch.float32)
    rng_noise=torch.Generator(device='cuda').manual_seed(20261010)
    noise=torch.randn(clean.shape,device='cuda',generator=rng_noise)
    uv=clean+job['noise']*clean.std(dim=0,unbiased=True)*noise
    scale=clean.square().mean().detach()
    candidate=torch.tensor(data['coordinates'],device='cuda',dtype=torch.float32)
    rng=torch.Generator(device='cuda').manual_seed(job['seed']+19)
    params=[p for name,p in model.named_parameters() if name!='log_lambda']
    pre,joint,lbmax=(10,10,3) if smoke else (3000,6000,400)
    batch,nf=(32,16) if smoke else (1024,384)
    history=[]
    def dl(indices):
        u,v,_=ns.velocity_fields(model,z_data[indices].detach().requires_grad_(True))
        return (torch.cat([u,v],1)-uv[indices]).square().mean()/scale
    def pl(z):
        fu,fv,_=ns.momentum_residual(model,z.detach().requires_grad_(True))
        return (fu.square().mean()+fv.square().mean())/2
    def record(stage,step,loss,data_loss,physics_loss):
        if step==1 or step%500==0:
            history.append(dict(stage=stage,step=step,loss=float(loss.detach()),data=float(data_loss.detach()),pde=float(physics_loss),coefficients=model.coefficients.detach().cpu().tolist()))
            print(job_id(job),stage,step,history[-1]['loss'],flush=True)
    model.log_lambda.requires_grad_(False)
    opt=torch.optim.Adam(params,lr=1e-3)
    for i in range(pre):
        idx=torch.randint(len(train),(batch,),device='cuda',generator=rng)
        opt.zero_grad(set_to_none=True)
        loss=dl(idx)
        assert torch.isfinite(loss)
        loss.backward(); opt.step()
        record('pretrain',i+1,loss,loss,0.)
    model.log_lambda.requires_grad_(True)
    opt=torch.optim.Adam([{'params':params,'lr':5e-4},{'params':[model.log_lambda],'lr':2e-3}])
    for i in range(joint):
        idx=torch.randint(len(train),(batch,),device='cuda',generator=rng)
        ids=torch.randint(len(candidate),(nf,),device='cuda',generator=rng)
        opt.zero_grad(set_to_none=True)
        a,b=dl(idx),pl(candidate[ids]); loss=a+b
        assert torch.isfinite(loss)
        loss.backward(); opt.step()
        record('joint',i+1,loss,a,b.detach())
    fixed_data=torch.randperm(len(train),device='cuda',generator=rng)[:32 if smoke else 2048]
    fixed_physics=candidate[torch.randperm(len(candidate),device='cuda',generator=rng)[:16 if smoke else 768]]
    opt=torch.optim.LBFGS(model.parameters(),lr=.8,max_iter=lbmax,max_eval=lbmax*2,history_size=50,line_search_fn='strong_wolfe',tolerance_grad=1e-9,tolerance_change=1e-11)
    calls=0
    def closure():
        nonlocal calls
        opt.zero_grad(set_to_none=True)
        a,b=dl(fixed_data),pl(fixed_physics); loss=a+b
        assert torch.isfinite(loss)
        loss.backward(); calls+=1
        record('lbfgs',calls,loss,a,b.detach())
        return loss
    assert float(closure().detach())==float(closure().detach())
    opt.step(closure)
    torch.save(model.state_dict(),folder/'final_weights.pt')
    original,p,pressure,aligned=ns.evaluate(model,data,test,'cuda')
    metric=dict(field_L2_percent=original['velocity_relative_L2_percent'],heldout_L2_percent=original['velocity_relative_L2_percent'],
                lambda1=original['lambda'][0],lambda2=original['lambda'][1],lambda1_error_percent=original['lambda_relative_error_percent'][0],lambda2_error_percent=original['lambda_relative_error_percent'][1],
                pressure_L2_percent=original['pressure_gauge_aligned_relative_L2_percent'],heldout_momentum_rms=original['held_out_momentum_rms'])
    arrays=dict(indices=test,coordinates=data['coordinates'][test],reference=data['velocity'][test],prediction=p,pressure=pressure,aligned_pressure=aligned)
    with (folder/'training_history.json').open('w',encoding='utf-8') as f:
        json.dump(history,f,ensure_ascii=False)
    np.savez_compressed(folder/'training_labels.npz',indices=train,clean=clean.cpu().numpy(),noisy=uv.cpu().numpy())
    return model,metric,arrays,dict(pretrain=pre,joint=joint,lbfgs_max_iter=lbmax,closure_calls=calls,batch=batch,nf=nf,reference_sha256=sha(path),split_sha256=sha(split_path),fixed_closure_verified=True,loss='velocity MSE / clean training velocity mean square + 0.5 mean(fu^2+fv^2)',truth_prior=False)


def self_test(mods):
    result=[]
    for case in CASES:
        counts={}
        for variant in ['MLP','Hybrid','MLP_matched','Fourier' if PRIMARY[case]=='Chebyshev' else 'Chebyshev','KAN_only']:
            seed_all(42)
            if case.startswith('NS'):
                data=np.load(CODE/'NSdiagnostics'/f'{NS_NAMES[case]}_diagnostic_data.npz')
                model=NSModel(data['bounds'],variant,[1.5,.02]).cuda().double()
                z=torch.rand(16,3,device='cuda',dtype=torch.float64,requires_grad=True)
                g,r=model.branch_outputs(z)
                assert torch.equal(model(z),g+r)
                fu,fv,div=mods[2].momentum_residual(model,z)
                loss=fu.square().mean()+fv.square().mean()
                assert div.abs().max()<1e-10
            else:
                model=ScalarModel(case,variant).cuda().double()
                x=torch.rand(16,1,device='cuda',dtype=torch.float64,requires_grad=True)
                t=torch.rand(16,1,device='cuda',dtype=torch.float64,requires_grad=True)
                g,r=model.branch_outputs(x,t)
                expected=g+r+(torch.cos(math.pi*x) if case=='KdV' else 0)
                assert torch.equal(model(x,t),expected)
                if case=='KdV':
                    assert torch.equal(model(x,t*0),torch.cos(math.pi*x))
                    loss=mods[1].compute_pde_residual(model,x,t).square().mean()
                else:
                    pts=tuple(a.double() for a in mods[0].sample_points(16,8,8,.5,'cuda'))
                    loss=mods[0].compute_loss(model,pts)[0]
            loss.backward()
            assert torch.isfinite(loss)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            counts[variant]=sum(p.numel() for p in model.parameters())
        assert abs(counts['MLP_matched']/counts['Hybrid']-1)<.01
        cross='Fourier' if PRIMARY[case]=='Chebyshev' else 'Chebyshev'
        assert counts[cross]==counts['Hybrid']
        result.append(dict(case=case,counts=counts,finite_high_order_backward=True,parallel_identity=True))
    save(OUT/'architecture_and_derivative_checks.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    global OUT
    if args.smoke:
        OUT=CODE/'auxiliary_suite_smoke_test2'
    OUT.mkdir(exist_ok=True)
    assert torch.cuda.is_available()
    mods=modules()
    self_test(mods)
    jobs=plan()
    save(OUT/'frozen_experiments.json',dict(jobs=jobs,seeds=SEEDS,smoke=args.smoke,
        protocol='Fixed audited labels/splits, paired initialization seeds, matched sampling stream within each seed, full original iteration budgets; no test-based selection.',
        interpretation='Basis family comparison uses identical trainable counts but is not a mathematical guarantee of low/high-frequency branch separation. Multiseed variability is conditional on the fixed label split.',
        code_sha256=sha(Path(__file__)),device=torch.cuda.get_device_name()))
    if args.self_test:
        return
    if args.smoke:
        jobs=[dict(case=c,variant=v,seed=42,noise=0.,init=[1.,.005] if c=='KdV' else [1.5,.02]) for c in CASES for v in ['Hybrid','MLP_matched','KAN_only','Fourier' if PRIMARY[c]=='Chebyshev' else 'Chebyshev']]
    if args.limit:
        jobs=jobs[:args.limit]
    for number,job in enumerate(jobs,1):
        folder=OUT/job_id(job)
        folder.mkdir(exist_ok=True)
        if (folder/'experiment_results.json').exists():
            old=json.loads((folder/'experiment_results.json').read_text(encoding='utf-8'))
            assert old['job']==job and old['smoke']==args.smoke
            assert old['code_sha256']==sha(Path(__file__)), 'Trainer changed: use a new versioned output root'
            print('PRESERVE',number,job_id(job),flush=True)
            continue
        print('START',number,len(jobs),job_id(job),flush=True)
        start=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with (folder/'execution_logs.txt').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):
            model,metric,arrays,config=(train_ns if job['case'].startswith('NS') else train_scalar)(job,folder,mods,args.smoke)
        torch.cuda.synchronize()
        weights=torch.load(folder/'final_weights.pt',map_location='cuda',weights_only=True)
        model.load_state_dict(weights,strict=True)
        # Re-evaluate the saved state, not an in-memory-only checkpoint.
        if job['case'].startswith('NS'):
            data=np.load(CODE/'NSdiagnostics'/f'{NS_NAMES[job["case"]]}_diagnostic_data.npz')
            check=mods[2].evaluate(model.float(),data,arrays['indices'],'cuda')[0]['velocity_relative_L2_percent']
        else:
            check=evaluate_scalar(model,arrays['x'],arrays['t'],arrays['reference'],arrays['heldout'])[0]['field_L2_percent']
        assert abs(check-metric['field_L2_percent'])<1e-9
        np.savez_compressed(folder/'final_prediction.npz',**arrays)
        result=dict(job=job,smoke=args.smoke,parameters=sum(p.numel() for p in model.parameters()),
                    seconds=time.perf_counter()-start,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
                    strict_reload_verified=True,code_sha256=sha(Path(__file__)),weights_sha256=sha(folder/'final_weights.pt'),config=config,**metric)
        save(folder/'experiment_results.json',result)
        print('DONE',number,len(jobs),json.dumps(result,ensure_ascii=False),flush=True)
        del model
        torch.cuda.empty_cache()
    save(OUT/'training_complete.json',dict(completed=len(jobs),smoke=args.smoke,code_sha256=sha(Path(__file__))))


if __name__=='__main__':
    main()
