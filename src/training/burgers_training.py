"""Isolated, paired-seed Burgers extension of the frozen variable-basis study."""
import argparse
import json
import math
from pathlib import Path
import platform

import numpy as np
import torch

import adaptive_basis as study
from forward_training import FROZEN_HASHES

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'adaptive_input_basis' / 'Burgersauxiliary_experiments'
REFERENCE = HERE / 'Burgersaudit' / 'Burgers_Cole_Hopfreference_on_same_grid.npz'
VARIANTS = ('MLP', 'Adaptive', 'SharedLearned', 'Uniform', 'Chebyshev', 'Fourier', 'RBF', 'Wavelet')
NU = .01 / math.pi


class BurgersModel(study.AdaptiveModel):
    def __init__(self, case, variant):
        assert case == 'Burgers' and variant in VARIANTS
        super().__init__('KdV', 'Adaptive' if variant == 'SharedLearned' else variant)
        self.case, self.variant = case, variant
        if variant == 'SharedLearned':
            # Same coefficients/RNG consumption as Adaptive; only gate sharing differs.
            self.kan[0].logits = torch.nn.Parameter(torch.zeros(1, len(study.FAMILIES)))

    def correction(self, z):
        q = torch.stack([z[:, 0], 2*z[:, 1]-1], 1)
        return self.mlp(q) + (self.kan(q) if self.variant != 'MLP' else 0.)

    def forward(self, z):
        return -torch.sin(math.pi*z[:, :1]) + z[:, 1:2]*self.correction(z)


def config(case, smoke=False):
    assert case == 'Burgers'
    return dict(adam_steps=6 if smoke else 8000, nf=16 if smoke else 2048,
                nb=8 if smoke else 128, lbfgs_max_iter=3 if smoke else 400,
                precision='FP32; TF32 off; FP64 final evaluation', internal_label_count=0,
                fixed_coefficients=[1., NU], boundary_weight=10., initial_condition='hard',
                sampling='uniform physical x in [-1,1], t in [0,1]',
                checkpoint='final only', smoke=smoke)


def digest():
    hashes = {name: study.base.sha(HERE/name) for name in FROZEN_HASHES}
    assert hashes == FROZEN_HASHES, 'Do not modify frozen studies'
    hashes[Path(__file__).name] = study.base.sha(Path(__file__))
    hashes['forward_training.py'] = study.base.sha(HERE/'forward_training.py')
    return hashes


def sample(case, count, bc_count, generator):
    assert case == 'Burgers'
    z = torch.rand(count, 2, device='cuda', generator=generator)
    z[:, 0] = 2*z[:, 0]-1
    return z.requires_grad_(), torch.rand(bc_count, 1, device='cuda', generator=generator)


def residual(model, z):
    u = model(z)
    du = study.base.derivative(u, z)
    uxx = study.base.derivative(du[:, :1], z)[:, :1]
    return du[:, 1:2] + u*du[:, :1] - NU*uxx


def loss_parts(model, z, bc):
    physics = residual(model, z).square().mean()
    left = torch.cat([-torch.ones_like(bc), bc], 1)
    right = torch.cat([torch.ones_like(bc), bc], 1)
    boundary = model(left).square().mean() + model(right).square().mean()
    return physics + 10*boundary, physics, boundary


def reference_test():
    from scipy.special import roots_hermite
    with np.load(REFERENCE) as ref:
        x, t, u = ref['x'].ravel(), ref['t'].ravel(), ref['usol']
    assert u.shape == (len(x), len(t)) and np.isfinite(u).all()
    assert np.all(np.diff(x)>0) and np.all(np.diff(t)>0)
    assert np.allclose([x[0],x[-1],t[0],t[-1]], [-1,1,0,1])
    assert np.max(np.abs(u[:,0]+np.sin(np.pi*x))) < 1e-12
    assert np.max(np.abs(u[[0,-1]])) < 1e-12
    # Recompute scattered sharp-gradient and regular points independently.
    ids = np.unique(np.linspace(0, len(x)-1, 21, dtype=int))
    tids = np.unique(np.linspace(1, len(t)-1, 13, dtype=int))
    xx, tt = np.meshgrid(x[ids], t[tids], indexing='ij')
    estimates = []
    for order in (256,512):
        nodes, weights = roots_hermite(order)
        valid = weights > 0
        y = xx.ravel()[:,None]-2*np.sqrt(NU*tt.ravel()[:,None])*nodes[valid]
        logw = np.log(weights[valid])[None,:]-np.cos(np.pi*y)/(2*np.pi*NU)
        w = np.exp(logw-logw.max(axis=1,keepdims=True))
        estimates.append((-np.sum(w*np.sin(np.pi*y),axis=1)/w.sum(axis=1)).reshape(xx.shape))
    delta = float(np.max(np.abs(estimates[1]-u[np.ix_(ids,tids)])))
    convergence = float(np.max(np.abs(estimates[0]-estimates[1])))
    assert max(delta,convergence)<1e-7
    return dict(sha256=study.base.sha(REFERENCE), grid=list(u.shape),
                independent_reference_max_error=delta, quadrature_difference=convergence)


def self_test():
    study.base.seed(142)
    z = torch.rand(9,2,device='cuda',dtype=torch.float64).requires_grad_()
    class Polynomial(torch.nn.Module):
        def forward(self, q):
            x,t = q[:,:1],q[:,1:2]
            return x*x+t*t+x*t
    x,t = z[:,:1],z[:,1:2]
    expected = 2*t+x+(x*x+t*t+x*t)*(2*x+t)-2*NU
    analytic_error = float((residual(Polynomial(),z)-expected).abs().max().detach())
    assert analytic_error<1e-12
    reports=[]
    paired={}
    for variant in VARIANTS:
        study.base.seed(42)
        model=BurgersModel('Burgers',variant).cuda().double()
        if variant in ('Adaptive','Uniform','SharedLearned'):
            paired[variant]={k:v.detach().clone() for k,v in model.state_dict().items() if not k.endswith('logits')}
        # Exercise every branch despite the deliberately zero initial KAN readout.
        with torch.no_grad():
            for p in model.parameters():
                p.add_(.003*torch.randn_like(p))
        gen=torch.Generator(device='cuda').manual_seed(142)
        z,bc=sample('Burgers',8,6,gen)
        z=z.double().detach().requires_grad_()
        bc=bc.double()
        total,_,_=loss_parts(model,z,bc)
        total.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        initial=z.detach().clone()
        initial[:,1]=0
        ic=float((model(initial)+torch.sin(math.pi*initial[:,:1])).abs().max().detach())
        assert ic<1e-12
        scales=torch.tensor([.7,.3],device='cuda',dtype=torch.float64)
        q=((z.detach()-.17)/scales).requires_grad_()
        u=model(q*scales+.17)
        du=study.base.derivative(u,q)/scales
        uxx=(study.base.derivative(du[:,:1],q)/scales)[:,:1]
        manual=du[:,1:2]+u*du[:,:1]-NU*uxx
        chain=float((manual-residual(model,z)).abs().max().detach())
        assert chain<1e-9
        gate=None
        if variant in ('Adaptive','SharedLearned'):
            gate=float(model.kan[0].logits.grad.abs().max())
            assert gate>1e-14
        reports.append(dict(variant=variant,parameters=sum(p.numel() for p in model.parameters()),
                            initial_error=ic,chain_error=chain,gate_gradient_max=gate,finite_backward=True))
    for variant in ('Uniform','SharedLearned'):
        assert paired['Adaptive'].keys()==paired[variant].keys()
        assert all(torch.equal(v,paired[variant][k]) for k,v in paired['Adaptive'].items())
    models=[]
    for variant in ('Adaptive','Uniform','SharedLearned'):
        study.base.seed(42)
        models.append(BurgersModel('Burgers',variant).cuda().double())
    for model in models[1:]:
        assert torch.equal(models[0].kan[0](z),model.kan[0](z))
        assert torch.equal(models[0](z),model(z))
    assert models[0].kan[0].logits.shape==(2,4) and models[2].kan[0].logits.shape==(1,4)
    q=z[:2].detach().requires_grad_()
    assert torch.autograd.gradcheck(models[2].kan[0],(q,),atol=1e-5,rtol=1e-3)
    def first(q):
        return study.base.derivative(models[2].kan[0](q),q)
    assert torch.autograd.gradgradcheck(first,(q,),atol=1e-5,rtol=1e-3)
    return dict(passed=True,models=reports,analytic_residual_error=analytic_error,
                identical_mixed_non_gate_initialization=True,reference=reference_test(),
                source_hashes=digest(),optimizer_steps=0)


def evaluate(model, folder, smoke):
    model.double()
    with np.load(REFERENCE) as ref:
        x,t,u=ref['x'].ravel(),ref['t'].ravel(),ref['usol']
    xx,tt=np.meshgrid(x,t,indexing='ij')
    z=np.stack([xx.ravel(),tt.ravel()],1)
    mask=z[:,1]>0
    z,target=z[mask],u.ravel()[mask,None]
    if smoke:
        z,target=z[::128],target[::128]
    predicted=[]
    with torch.no_grad():
        for start in range(0,len(z),2048):
            batch=torch.tensor(z[start:start+2048],device='cuda',dtype=torch.float64)
            predicted.append(model(batch).cpu().numpy())
    predicted=np.concatenate(predicted)
    np.savez_compressed(folder/'final_prediction.npz',coordinates=z,reference=target,prediction=predicted)
    gen=torch.Generator(device='cuda').manual_seed(20261017)
    zz,bb=sample('Burgers',64 if smoke else 4096,32 if smoke else 512,gen)
    _,p,b=loss_parts(model,zz.double().detach().requires_grad_(),bb.double())
    return dict(field_L2_percent_excluding_initial_time=float(100*np.linalg.norm(predicted-target)/np.linalg.norm(target)),
                independent_physics_rms=float(p.detach().sqrt()), independent_boundary_loss=float(b.detach()),
                reference_sha256=study.base.sha(REFERENCE),internal_label_count=0,fixed_coefficients=[1.,NU])


def gate_diagnostic(model):
    if model.variant not in ('Adaptive','Uniform','SharedLearned'):
        return None
    return dict(variables=['shared_x_t'] if model.variant=='SharedLearned' else ['x','t'],
                families=study.FAMILIES,weights=model.kan[0].weights().detach().cpu().tolist(),
                warning='Soft gates can compensate with free coefficients; not identifiable physical importance.')


def install():
    # This affects only the new worker process, never the frozen files or live workers.
    study.AdaptiveModel=BurgersModel
    study.config=config
    study.digest=digest
    study.gate_diagnostic=gate_diagnostic
    study.base.sample=sample
    study.base.loss_parts=loss_parts
    study.base.evaluate=evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--node',choices=['A','B','C'],required=True)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    assert platform.system()=='Linux', 'Cloud only; no local training'
    assert torch.cuda.is_available() and torch.__version__=='2.5.1+cu121'
    root=ROOT/('smoke_'+args.node if args.smoke else args.node)
    root.mkdir(parents=True,exist_ok=True)
    hashes=digest()
    if args.self_test:
        study.base.save(root/'self_test.json',self_test())
        print('BURGERS_SELF_TEST_PASSED',flush=True)
        return
    test=json.loads((ROOT/args.node/'self_test.json').read_text(encoding='utf-8'))
    assert test['passed'] and test['source_hashes']==hashes
    assert test['reference']['sha256']==study.base.sha(REFERENCE)
    seed={'A':42,'B':43,'C':44}[args.node]
    variants=('Adaptive','SharedLearned') if args.smoke else VARIANTS
    jobs=[dict(case='Burgers',variant=v,seed=seed) for v in variants]
    manifest=dict(jobs=jobs,source_hashes=hashes,reference_sha256=study.base.sha(REFERENCE),config=config('Burgers',args.smoke),
        seeds=[42,43,44],all_seeds_reported=True,architecture='Same scalar architecture as current KdV; first KAN layer only varies',
        paired_initialization='Adaptive, SharedLearned and Uniform have identical non-gate initial tensors',
        limitations=['Equal steps and approximate parameters, not equal time', 'Hard IC differs from historical soft-IC Burgers protocol',
                     'Only initial KAN layer adapts; hidden KAN remains Chebyshev', 'Gate weights are not identifiable physical importance'])
    import fcntl
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'manifest.json'
        if path.exists():
            assert json.loads(path.read_text(encoding='utf-8'))==manifest
        else:
            study.base.save(path,manifest)
        install()
        results=[]
        for job in jobs:
            try:
                result=study.train(job,root,args.smoke)
            except Exception as exc:
                study.base.save(root/'progress.json',dict(status='failed',job=job,error=repr(exc)))
                raise
            results.append(result)
            study.base.save(root/'completed.json',results)
            torch.cuda.empty_cache()
        study.base.save(root/'progress.json',dict(status='complete',completed=len(results),jobs=len(jobs)))


if __name__=='__main__':
    main()
