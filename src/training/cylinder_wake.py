"""Current-model cylinder-wake inverse conditions, without periodic constraints."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import experiment_conditions as common

study, inverse, base = common.study, common.inverse, common.study.base
HERE = Path(__file__).resolve().parent
DATA = HERE / 'paper_wake_data'
ORIGINAL = HERE / 'NSdiagnostics'

def jobs():
    groups = [('wake_clean','clean'),('wake_noise5','noise_5pct'),('wake_init','alternative_initialization')]
    return [dict(id=f'paper_{g}_{v}_seed{s}',kind='paper',group=g,condition=c,mode='inverse',
                 job=dict(case='NS_wake',variant=v,seed=s))
            for g,c in groups for v in common.MODELS for s in (42,43,44)]

def prepare_data():
    source=ORIGINAL/'cylinder_wake_diagnostic_data.npz'
    split=ORIGINAL/'cylinder_wake_fixed_split.npz'
    with np.load(source) as data, np.load(split) as indices:
        train,test=indices['train'],indices['test']
        assert len(train)==10000 and len(test)==20000 and not np.intersect1d(train,test).size
        z,y=data['coordinates'],data['velocity']
        assert z.shape[1]==3 and y.shape[1]==2 and np.isfinite(z).all() and np.isfinite(y).all()
        meta=dict(source_sha256=base.sha(source),split_sha256=base.sha(split),
            lower=z.min(0).tolist(),upper=z.max(0).tolist(),training_points=len(train),test_points=len(test),
            no_periodic_boundary=True,no_hard_initial_field=True)
        DATA.mkdir(exist_ok=True)
        target=DATA/'observations.npz'
        if target.exists():
            with np.load(target) as old:
                assert np.array_equal(old['coordinates'],z[train]) and np.array_equal(old['target'],y[train])
        else:np.savez_compressed(target,coordinates=z[train],target=y[train],indices=train)
        path=DATA/'metadata.json'
        if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==meta
        else:base.save(path,meta)
    return meta

def observation_arrays(spec):
    with np.load(DATA/'observations.npz') as data:
        z,clean,indices=data['coordinates'].copy(),data['target'].copy(),data['indices'].copy()
    y=clean.copy();mask=z[:,-1]>0;scales=clean[mask].astype(np.float64).std(0,ddof=0)
    if spec['condition']=='noise_5pct':
        rng=np.random.default_rng(20281018+spec['job']['seed'])
        y[mask]+=(.05*scales*rng.standard_normal(y[mask].shape)).astype(y.dtype)
    assert np.array_equal(y[~mask],clean[~mask])
    return z,y,dict(coordinates_sha256=common.array_sha(z),labels_sha256=common.array_sha(y),
        indices_sha256=common.array_sha(indices),noise_component_std=scales.tolist(),
        noise_relative_std=.05 if spec['condition']=='noise_5pct' else 0.)

class Network(common.BASE_MODEL):
    def __init__(self,case,variant):
        assert case=='NS_wake'
        mapped='Adaptive' if variant=='SharedLearned' else 'Fourier' if variant=='Fourier_HFourier' else variant
        super().__init__('NS_periodic',mapped)
        self.case,self.variant=case,variant
        if variant=='SharedLearned':self.kan[0].logits=nn.Parameter(self.kan[0].logits[:1].detach().clone())
        if variant=='Fourier_HFourier':self.kan[2]=common.hidden_layer(self.kan[2],'Fourier')
        meta=json.loads((DATA/'metadata.json').read_text(encoding='utf-8'))
        self.register_buffer('domain_lower',torch.tensor(meta['lower'],dtype=torch.float32))
        self.register_buffer('domain_upper',torch.tensor(meta['upper'],dtype=torch.float32))
    def correction(self,z):
        q=2*(z-self.domain_lower)/(self.domain_upper-self.domain_lower)-1
        return self.mlp(q)+(self.kan(q) if self.variant!='MLP' else 0.)
    def forward(self,z):return self.correction(z)

def sample(case,count,bc_count,generator):
    assert case=='NS_wake'
    meta=json.loads((DATA/'metadata.json').read_text(encoding='utf-8'))
    lower=torch.tensor(meta['lower'],device='cuda');upper=torch.tensor(meta['upper'],device='cuda')
    z=lower+torch.rand(count,3,device='cuda',generator=generator)*(upper-lower)
    return z.requires_grad_(),torch.empty(0,2,device='cuda')

def parts(model,z,bc,zd,yd,scale):
    data=inverse.observation_loss(model,zd,yd,scale)
    physics=inverse.residual(model,z).square().mean()
    return data+physics,data,physics,physics.detach()*0

def evaluate(model,folder,smoke):
    model.double()
    source=ORIGINAL/'cylinder_wake_diagnostic_data.npz'
    with np.load(source) as ref,np.load(ORIGINAL/'cylinder_wake_fixed_split.npz') as split:
        z,target,truth=ref['coordinates'],ref['velocity'],ref['truth']
        test=split['test']
    selected=np.flatnonzero(z[:,-1]>0)
    if smoke:selected=selected[::128]
    pred=[]
    for start in range(0,len(selected),1024):
        zz=torch.tensor(z[selected[start:start+1024]],device='cuda',dtype=torch.float64,requires_grad=True)
        pred.append(inverse.prediction(model,zz).detach().cpu().numpy())
    pred=np.concatenate(pred);target=target[selected];heldout=np.isin(selected,test)
    assert heldout.any()
    rng=torch.Generator(device='cuda').manual_seed(20261018)
    zz,_=sample('NS_wake',64 if smoke else 1024,0,rng)
    residual=inverse.residual(model,zz.double().detach().requires_grad_())
    coeff=model.coefficients.detach().cpu().numpy()
    np.savez_compressed(folder/'final_prediction.npz',indices=selected,coordinates=z[selected],
        reference=target,prediction=pred,heldout=heldout)
    relative=lambda a,b:float(100*np.linalg.norm(a-b)/np.linalg.norm(b))
    gates=None
    if model.variant in ('Adaptive','Uniform','SharedLearned'):
        gates=dict(weights=model.kan[0].weights().detach().cpu().tolist(),families=study.FAMILIES,
                   warning='Gates are not identifiable causal importance')
    return dict(heldout_L2_percent=relative(pred[heldout],target[heldout]),heldout_points=int(heldout.sum()),
        field_L2_percent_excluding_initial_time=relative(pred,target),coefficients=coeff.tolist(),
        reference_coefficients=truth.tolist(),lambda1_estimated=True,
        lambda1_error_percent=float(abs(coeff[0]/truth[0]-1)*100),
        lambda2_error_percent=float(abs(coeff[1]/truth[1]-1)*100),
        independent_physics_rms=float(residual.square().mean().sqrt().detach()),independent_boundary_loss=None,
        reference_sha256=base.sha(source),gate_diagnostic=gates,no_periodic_boundary=True,no_hard_initial_field=True)

def install(spec):
    assert common.INSTALLED is None and spec in jobs()
    common.INSTALLED=spec
    hashes=common.INV_DIGEST()
    for name in ('experiment_conditions.py',Path(__file__).name,'architecture_controls.py'):
        hashes[name]=base.sha(HERE/name)
    z,y,meta=observation_arrays(spec)
    inverse.INITIAL['NS_wake']=[.5,.005] if spec['condition']=='alternative_initialization' else [1.5,.02]
    study.AdaptiveModel=Network
    def config(case,smoke=False):
        value=common.INV_CONFIG(case,smoke)
        return dict(value,hard_initial_condition=False,loss='normalized velocity observation MSE + unforced momentum MSE',
            periodic_boundary=False,nb=0,paper_condition=spec['condition'],observation_transform=meta,
            noise=.05 if spec['condition']=='noise_5pct' else 0.)
    def loader(case,device):
        zz=torch.tensor(z,device=device,dtype=torch.float32)
        yy=torch.tensor(y,device=device,dtype=torch.float32)
        return zz,yy,yy.square().mean().detach()
    inverse.config=config;inverse.load_observations=loader
    inverse.digest=lambda:dict(hashes)
    inverse.data_hashes=lambda case:dict(observations=base.sha(DATA/'observations.npz'),
        metadata=base.sha(DATA/'metadata.json'),split=base.sha(ORIGINAL/'cylinder_wake_fixed_split.npz'),transform=meta)
    inverse.parts=parts;inverse.evaluate=evaluate;base.sample=sample
    return inverse

def self_test(spec,device='cpu'):
    torch.set_num_threads(1);base.seed(spec['job']['seed'])
    model=inverse.InverseModel('NS_wake',spec['job']['variant']).to(device).double()
    with torch.no_grad():
        for p in model.network.parameters():p.add_(.003*torch.randn_like(p))
    lower,upper=model.network.domain_lower,model.network.domain_upper
    z=(lower+(upper-lower)*torch.rand(5,3,device=device,dtype=torch.float64)).requires_grad_()
    scales=torch.tensor([.7,1.3,.3],device=device,dtype=torch.float64)
    q=((z.detach()-.17)/scales).requires_grad_();pred=model(q*scales+.17)
    grad=lambda value:base.derivative(value,q)/scales
    ds,dp=grad(pred[:,:1]),grad(pred[:,1:2]);u,v=ds[:,1:2],-ds[:,:1]
    du,dv=grad(u),grad(v)
    lu=grad(du[:,:1])[:,:1]+grad(du[:,1:2])[:,1:2]
    lv=grad(dv[:,:1])[:,:1]+grad(dv[:,1:2])[:,1:2]
    l1,l2=model.coefficients
    manual=torch.cat([du[:,2:3]+l1*(u*du[:,:1]+v*du[:,1:2])+dp[:,:1]-l2*lu,
                      dv[:,2:3]+l1*(u*dv[:,:1]+v*dv[:,1:2])+dp[:,1:2]-l2*lv],1)
    direct=inverse.residual(model,z)
    error=float((manual-direct).abs().max().detach());assert error<1e-9
    direct.square().mean().backward()
    # Constant streamfunction/pressure readout offsets disappear from velocity
    # and momentum observations. Their absent gradients are gauge freedoms.
    gauge_ids={id(model.network.mlp[-1].bias)}
    if model.variant!='MLP':gauge_ids.add(id(model.network.kan[-1].bias))
    inactive=[]
    for name,p in model.named_parameters():
        if p.grad is None:
            assert id(p) in gauge_ids,name
            inactive.append(name)
        else:assert torch.isfinite(p.grad).all(),name
    assert torch.all(model.log_coefficients.grad.abs()>1e-14)
    restored=inverse.InverseModel('NS_wake',spec['job']['variant']).to(device).double()
    restored.load_state_dict(model.state_dict());assert torch.equal(model(z),restored(z))
    with np.load(DATA/'observations.npz') as obs,np.load(ORIGINAL/'cylinder_wake_fixed_split.npz') as split:
        assert not np.intersect1d(obs['indices'],split['test']).size
    return dict(passed=True,spec=spec,device=device,optimizer_steps=0,physical_derivative_max_difference=error,
        finite_backward=True,reload_exact=True,no_periodic_boundary=True,no_hard_initial_field=True,
        unobservable_constant_offsets=inactive,source_sha256=base.sha(Path(__file__)))

train=common.train

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-data',action='store_true')
    parser.add_argument('--task');parser.add_argument('--device',default='cpu');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.prepare_data:print(json.dumps(prepare_data()))
    else:
        spec=next(t for t in jobs() if t['id']==args.task)
        install(spec);base.save(args.output,self_test(spec,args.device))
