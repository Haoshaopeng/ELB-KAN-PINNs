"""Four new PDE families, two predeclared regimes, nine models, three seeds.

No interior solution labels in training. Analytic solutions and an independently
grid-checked Allen-Cahn reference are opened only after final weights are saved.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn
from architecture_controls import hidden_values

HERE=Path(__file__).resolve().parent
DATA=HERE/'generalization_data'
FAMILIES=('Chebyshev','Fourier','RBF','Wavelet')
MODELS=('MLP','Chebyshev','Fourier','RBF','Wavelet','Uniform','Adaptive','SharedLearned','Fourier_HFourier')
CASES=('Heat2D','AllenCahn','Helmholtz2D','Wave1D')
SEEDS=(42,43,44)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(temp,path)

def jobs():
    return [dict(id=f'new_{case}_{regime}_{model}_seed{seed}',kind='newpde',case=case,regime=regime,variant=model,seed=seed)
            for case in CASES for regime in ('base','stress') for model in MODELS for seed in SEEDS]

def physics(case,regime):
    stress=regime=='stress'
    if case=='Heat2D':return dict(ax=.1,ay=.001 if stress else .01,modes=[(1,5 if stress else 2,1.),(5 if stress else 3,1,.25)])
    if case=='AllenCahn':return dict(epsilon=.0001 if stress else .001,reaction=5.)
    if case=='Helmholtz2D':return dict(k=15. if stress else 5.,m=4 if stress else 1,n=5 if stress else 2)
    if case=='Wave1D':return dict(c=2. if stress else 1.,modes=[(3 if stress else 1,1.),(7 if stress else 3,.3)])
    raise ValueError(case)

def config(job,smoke=False):
    return dict(case=job['case'],regime=job['regime'],physics=physics(job['case'],job['regime']),
                adam_steps=2 if smoke else 8000,lbfgs_max_iter=2 if smoke else 400,
                nf=16 if smoke else 1024,lbfgs_nf=32 if smoke else 2048,
                lr=.001,gradient_clip=1.,precision='float32',evaluation_precision='float64',
                kan_widths=[32,32],input_terms=16,hidden_terms=16,mlp_hidden=[40,40,40],
                mlp_baseline=[64]*6,loss='PDE mean square; hard IC/BC, no interior labels',
                checkpoint='final only',sampling='paired random interior, independent of model initialization',
                input_mapping='unit-domain coordinates; Fourier k*pi*s for Dirichlet harmonics',
                normalization='input fixed quadrature; hidden none',smoke=smoke)

def dimension(case):return 3 if case=='Heat2D' else 2
def unit(z,case):
    if case in ('AllenCahn','Wave1D'):return torch.cat([(z[:,:1]+1)/2,z[:,1:]],1)
    return z

def values(q,family,terms):
    if family=='Fourier':
        freq=torch.arange(1,terms//2+1,device=q.device,dtype=q.dtype)
        phase=math.pi*(q.unsqueeze(-1)+1)/2*freq
        return torch.cat([phase.cos(),phase.sin()],-1)
    return hidden_values(q,family,terms)

class InputLayer(nn.Module):
    def __init__(self,inp,variant):
        super().__init__()
        self.families=FAMILIES if variant in ('Uniform','Adaptive','SharedLearned') else (variant,)
        self.terms=16//len(self.families)
        self.coeff=nn.Parameter(torch.randn(inp,32,16)*.01)
        gate=torch.zeros(1 if variant=='SharedLearned' else inp,len(self.families))
        if variant in ('Adaptive','SharedLearned'):self.logits=nn.Parameter(gate)
        else:self.register_buffer('logits',gate)
        q=torch.linspace(-1,1,2049,dtype=torch.float64)[:,None].expand(-1,inp)
        f=torch.cat([values(q,v,self.terms) for v in self.families],-1)
        mu=f.mean(0);sd=(f-mu).square().mean(0).sqrt();constant=sd<1e-10
        self.register_buffer('means',torch.where(constant,0,mu).float())
        self.register_buffer('scales',torch.where(constant,1,sd).float())
    def forward(self,q):
        f=torch.cat([values(q,v,self.terms) for v in self.families],-1)
        weights=(len(self.families)*self.logits.softmax(-1)).repeat_interleave(self.terms,-1)
        return torch.einsum('nik,iok->no',(f-self.means)/self.scales*weights,self.coeff)

class HiddenLayer(nn.Module):
    def __init__(self,family):
        super().__init__();self.family=family
        self.coeff=nn.Parameter(torch.randn(32,32,16)*.01)
    def forward(self,x):return torch.einsum('nik,iok->no',hidden_values(x,self.family,16),self.coeff)

def mlp(inp,widths):
    layers=[]
    for width in widths:
        linear=nn.Linear(inp,width);nn.init.xavier_normal_(linear.weight);nn.init.zeros_(linear.bias)
        layers += [linear,nn.Tanh()];inp=width
    final=nn.Linear(inp,1);nn.init.xavier_normal_(final.weight);nn.init.zeros_(final.bias)
    return nn.Sequential(*layers,final)

def initial(z,case,p):
    x=z[:,:1]
    if case=='AllenCahn':return x.square()*torch.cos(math.pi*x)
    if case=='Wave1D':return sum(a*torch.sin(m*math.pi*(x+1)/2) for m,a in p['modes'])
    y=z[:,1:2]
    return sum(a*torch.sin(m*math.pi*x)*torch.sin(n*math.pi*y) for m,n,a in p['modes'])

class Model(nn.Module):
    def __init__(self,job):
        super().__init__();self.job=job;self.case=job['case'];self.p=physics(self.case,job['regime'])
        kind=job['variant'];inp=dimension(self.case)
        self.mlp=mlp(inp,[64]*6 if kind=='MLP' else [40]*3)
        if kind!='MLP':
            self.kan=nn.Sequential(InputLayer(inp,'Fourier' if kind=='Fourier_HFourier' else kind),nn.Tanh(),
                HiddenLayer('Fourier' if kind=='Fourier_HFourier' else 'Chebyshev'),nn.Tanh(),nn.Linear(32,1))
            nn.init.zeros_(self.kan[-1].weight);nn.init.zeros_(self.kan[-1].bias)
    def forward(self,z):
        q=2*unit(z,self.case)-1
        correction=self.mlp(q)+(self.kan(q) if hasattr(self,'kan') else 0.)
        x=z[:,:1]
        if self.case=='Helmholtz2D':
            y=z[:,1:2];return 16*x*(1-x)*y*(1-y)*correction
        base=initial(z,self.case,self.p);t=z[:,-1:]
        if self.case=='Heat2D':
            y=z[:,1:2];factor=16*x*(1-x)*y*(1-y)
        else:factor=1-x.square()
        return base+(t.square() if self.case=='Wave1D' else t)*factor*correction

def grad(y,z):return torch.autograd.grad(y,z,torch.ones_like(y),create_graph=True)[0]

def source(z,p):
    x,y=z[:,:1],z[:,1:2]
    return sum(a*(math.pi**2*(m*m+n*n)-p['k']**2)*torch.sin(m*math.pi*x)*torch.sin(n*math.pi*y)
               for m,n,a in [(p['m'],p['n'],1.),(p['m']+1,p['n']+2,.25)])

def residual(model,z):
    u=model(z);d=grad(u,z);xx=grad(d[:,:1],z)[:,:1];p=model.p
    if model.case=='AllenCahn':return d[:,1:2]-p['epsilon']*xx+p['reaction']*(u**3-u)
    if model.case=='Wave1D':return grad(d[:,1:2],z)[:,1:2]-p['c']**2*xx
    yy=grad(d[:,1:2],z)[:,1:2]
    if model.case=='Heat2D':return d[:,2:3]-p['ax']*xx-p['ay']*yy
    # The fixed scale uses only equation coefficients, not the unknown field.
    return (-xx-yy-p['k']**2*u-source(z,p))/max(1.,p['k']**2)

def exact(z,case,p):
    x=z[:,:1]
    if case=='Wave1D':
        return sum(a*torch.sin(m*math.pi*(x+1)/2)*torch.cos(p['c']*m*math.pi*z[:,1:2]/2) for m,a in p['modes'])
    y=z[:,1:2]
    if case=='Heat2D':
        t=z[:,2:3]
        return sum(a*torch.sin(m*math.pi*x)*torch.sin(n*math.pi*y)*torch.exp(-math.pi**2*(p['ax']*m*m+p['ay']*n*n)*t) for m,n,a in p['modes'])
    if case=='Helmholtz2D':return torch.sin(p['m']*math.pi*x)*torch.sin(p['n']*math.pi*y)+.25*torch.sin((p['m']+1)*math.pi*x)*torch.sin((p['n']+2)*math.pi*y)
    raise ValueError('Allen-Cahn uses the independently converged reference')

def make_reference(case,regime):
    path=DATA/(case+'_'+regime+'.npz')
    if path.exists():return
    p=physics(case,regime);evidence=dict(case=case,regime=regime,physics=p,optimizer_steps=0)
    if case=='AllenCahn':
        from scipy.integrate import solve_ivp
        from scipy.sparse import diags
        times=np.linspace(0,1,101)
        solutions=[]
        for intervals in (1024,2048,4096):
            x=np.linspace(-1,1,intervals+1);dx=x[1]-x[0];nu=p['epsilon'];r=p['reaction']
            operator=diags([np.full(intervals-2,nu/dx**2),np.full(intervals-1,-2*nu/dx**2),np.full(intervals-2,nu/dx**2)],[-1,0,1],format='csc')
            boundary=np.zeros(intervals-1);boundary[[0,-1]]=-nu/dx**2
            def rhs(t,u):return operator@u+boundary-r*(u**3-u)
            def jac(t,u):return operator+diags(-r*(3*u*u-1),0,format='csc')
            sol=solve_ivp(rhs,(0.,1.),x[1:-1]**2*np.cos(math.pi*x[1:-1]),method='BDF',jac=jac,t_eval=times,rtol=1e-9,atol=1e-11)
            assert sol.success and np.isfinite(sol.y).all()
            field=np.vstack([-np.ones((1,len(times))),sol.y,-np.ones((1,len(times)))])
            solutions.append(field[::intervals//256])
        coarse=np.linalg.norm(solutions[0]-solutions[1])/np.linalg.norm(solutions[1])
        fine=np.linalg.norm(solutions[1]-solutions[2])/np.linalg.norm(solutions[2])
        assert fine < 1e-3 and fine < coarse, (coarse,fine)
        xx,tt=np.meshgrid(np.linspace(-1,1,257),times[1:],indexing='ij')
        z=np.stack([xx.ravel(),tt.ravel()],1);truth=solutions[-1][:,1:].reshape(-1,1)
        evidence.update(method='BDF, second-order finite differences, exact Dirichlet boundary',intervals=[1024,2048,4096],coarse_fine_relative=coarse,fine_finest_relative=fine,
                        rtol=1e-9,atol=1e-11,reference_validated=True)
    else:
        axes=([np.linspace(0,1,49),np.linspace(0,1,49),np.linspace(0,1,41)[1:]] if case=='Heat2D' else
              [np.linspace(0,1,129),np.linspace(0,1,129)] if case=='Helmholtz2D' else
              [np.linspace(-1,1,257),np.linspace(0,1,101)[1:]])
        z=np.stack([v.ravel() for v in np.meshgrid(*axes,indexing='ij')],1)
        truth=exact(torch.tensor(z,dtype=torch.float64),case,p).numpy()
        evidence.update(method='analytic solution',reference_validated=True)
    DATA.mkdir(exist_ok=True)
    np.savez_compressed(path,coordinates=z,target=truth)
    evidence['sha256']=sha(path);save(path.with_suffix('.json'),evidence)

def sample(job,n,rng,device):
    z=torch.rand(n,dimension(job['case']),generator=rng,device=device)
    if job['case'] in ('AllenCahn','Wave1D'):z[:,0]=2*z[:,0]-1
    return z.requires_grad_()

def self_test(job,device='cpu'):
    torch.set_num_threads(1);torch.manual_seed(job['seed'])
    model=Model(job).to(device).double()
    with torch.no_grad():
        for p in model.parameters():p.add_(.002*torch.randn_like(p))
    z=torch.rand(5,dimension(model.case),device=device,dtype=torch.float64)
    if model.case in ('AllenCahn','Wave1D'):z[:,0]=2*z[:,0]-1
    z.requires_grad_();u=model(z);r=residual(model,z)
    assert torch.isfinite(r).all()
    r.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    scale=torch.linspace(.4,.9,z.shape[1],device=device,dtype=z.dtype)
    q=((z.detach()-.1)/scale).requires_grad_()
    a,b=model(z),model(q*scale+.1);errors=[]
    for axis in range(z.shape[1]):
        da,db=a,b
        for _ in range(3):
            da=grad(da,z)[:,axis:axis+1];db=grad(db,q)[:,axis:axis+1]/scale[axis]
            assert torch.allclose(da,db,rtol=1e-7,atol=1e-8)
            errors.append(float((da-db).abs().max().detach()))
    if model.case!='Helmholtz2D':
        zi=z.detach().clone();zi[:,-1]=0;zi.requires_grad_()
        assert torch.allclose(model(zi),initial(zi,model.case,model.p),atol=1e-12,rtol=0)
        if model.case=='Wave1D':assert grad(model(zi),zi)[:,1].abs().max()<1e-12
    for axis in range(2 if model.case in ('Heat2D','Helmholtz2D') else 1):
        for edge in ((-1.,1.) if model.case in ('AllenCahn','Wave1D') else (0.,1.)):
            zb=z.detach().clone();zb[:,axis]=edge
            truth=-torch.ones_like(model(zb)) if model.case=='AllenCahn' else torch.zeros_like(model(zb))
            assert torch.allclose(model(zb),truth,atol=1e-12,rtol=0)
    exact_error=None
    if model.case!='AllenCahn':
        class Exact(nn.Module):
            def __init__(self):super().__init__();self.case=model.case;self.p=model.p
            def forward(self,z):return exact(z,self.case,self.p)
        exact_error=float(residual(Exact(),z).abs().max().detach())
        assert exact_error<1e-9
    return dict(passed=True,job=job,optimizer_steps=0,physical_chain_error=max(errors),exact_pde_residual=exact_error,
                parameters=sum(p.numel() for p in model.parameters()),hard_conditions=True)

def evaluate(model,root,job):
    model.double().eval()
    path=DATA/(job['case']+'_'+job['regime']+'.npz')
    with np.load(path) as a:z,y=a['coordinates'],a['target']
    chunks=[]
    with torch.no_grad():
        for i in range(0,len(z),1024):chunks.append(model(torch.tensor(z[i:i+1024],device='cuda',dtype=torch.float64)).cpu().numpy())
    pred=np.concatenate(chunks);error=100*np.linalg.norm(pred-y)/np.linalg.norm(y)
    rng=torch.Generator(device='cuda').manual_seed(20261018)
    test=sample(job,1024,rng,'cuda').detach().double().requires_grad_()
    rms=float(residual(model,test).square().mean().sqrt().detach())
    slices=[]
    if job['case']!='Helmholtz2D':
        for t in np.unique(z[:,-1]):
            sel=z[:,-1]==t
            slices.append(dict(time=float(t),L2_percent=float(100*np.linalg.norm(pred[sel]-y[sel])/max(np.linalg.norm(y[sel]),1e-30))))
    np.savez_compressed(root/'prediction.npz',coordinates=z,reference=y,prediction=pred)
    gates=model.kan[0].logits.softmax(-1).detach().cpu().tolist() if hasattr(model,'kan') else None
    return dict(field_L2_percent=float(error),independent_scaled_pde_rms=rms,time_slices=slices,
                reference_sha256=sha(path),gates=gates,gate_warning='Weights are not causal physical importance')

def train(job,root,smoke=False):
    assert platform.system()=='Linux' and torch.cuda.is_available(), 'Cloud training only'
    assert torch.__version__=='2.5.1+cu121'
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    assert not (root/'started.json').exists(), 'No overwrite or implicit retry'
    cfg=config(job,smoke);torch.set_num_threads(1);torch.manual_seed(job['seed']);torch.cuda.manual_seed_all(job['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=Model(job).cuda();rng=torch.Generator(device='cuda').manual_seed(job['seed']+1000)
    save(root/'started.json',dict(job=job,config=cfg,source_sha256=sha(__file__),time=time.time()))
    history=[];start=time.time()
    def record(stage,step,loss):
        row=dict(stage=stage,step=step,loss=float(loss.detach()),elapsed_seconds=time.time()-start)
        history.append(row);save(root/'history.json',history);save(root/'progress.json',dict(job=job,config=cfg,**row))
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
    scheduler=torch.optim.lr_scheduler.StepLR(opt,max(1,cfg['adam_steps']//2),gamma=.5)
    for step in range(1,cfg['adam_steps']+1):
        z=sample(job,cfg['nf'],rng,'cuda');opt.zero_grad(set_to_none=True)
        loss=residual(model,z).square().mean();assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();scheduler.step()
        if step==1 or step%100==0 or step==cfg['adam_steps']:record('Adam',step,loss)
        if step%1000==0:torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=rng.get_state(),step=step),root/'checkpoint.pt')
    z=sample(job,cfg['lbfgs_nf'],rng,'cuda')
    opt=torch.optim.LBFGS(model.parameters(),lr=.8,max_iter=cfg['lbfgs_max_iter'],max_eval=2*cfg['lbfgs_max_iter'],history_size=50,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-10)
    calls=0
    def closure():
        nonlocal calls
        opt.zero_grad(set_to_none=True);loss=residual(model,z).square().mean();assert torch.isfinite(loss)
        loss.backward();calls+=1
        if calls==1 or calls%25==0:record('L-BFGS',calls,loss)
        return loss
    assert float(closure().detach())==float(closure().detach())
    opt.step(closure)
    torch.save(model.state_dict(),root/'final.pt');torch.cuda.synchronize()
    result=dict(job=job,config=cfg,weights_sha256=sha(root/'final.pt'),source_sha256=sha(__file__),
                parameters=sum(p.numel() for p in model.parameters()),training_seconds=time.time()-start,closure_calls=calls,
                timing_warning='Concurrent wall time is not isolated GPU efficiency',metrics=evaluate(model,root,job))
    save(root/'result.json',result);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['references','self-tests']);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.action=='references':
        for case in CASES:
            for regime in ('base','stress'):
                make_reference(case,regime);print('REFERENCE_READY',case,regime,flush=True)
    else:
        reports=[self_test(job) for job in jobs() if job['seed']==42]
        save(args.out,dict(passed=True,reports=reports,optimizer_steps=0,source_sha256=sha(__file__)))
        print('SELF_TESTS_PASSED',len(reports),flush=True)
