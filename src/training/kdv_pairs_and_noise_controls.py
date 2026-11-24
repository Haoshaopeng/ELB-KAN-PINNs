"""Only the 33 missing KdV pairs and 12 Fourier robustness controls."""
import hashlib
from pathlib import Path

FAMILIES=('Chebyshev','Fourier','RBF','Wavelet')
SEEDS=(42,43,44)
EXISTING={(f,'Chebyshev') for f in FAMILIES}|{('Fourier','Fourier')}

def jobs():
    result=[]
    for first in FAMILIES:
        for hidden in FAMILIES:
            if (first,hidden) in EXISTING:continue
            for seed in SEEDS:
                variant=first+'_H'+hidden
                result.append(dict(id='pair_KdV_'+variant+'_seed'+str(seed),kind='pair',
                    input_variant=first,hidden_basis=hidden,job=dict(case='KdV',variant=variant,seed=seed)))
    for condition in ('noise_1pct','observations_25pct'):
        for case in ('KdV','NS_periodic'):
            for seed in SEEDS:
                result.append(dict(id=condition+'_'+case+'_Fourier_seed'+str(seed),kind='robust',
                    condition=condition,job=dict(case=case,variant='Fourier',seed=seed)))
    assert len(result)==len({j['id'] for j in result})==45
    return result

def install(spec):
    import numpy as np
    import torch
    import adaptive_basis as study
    import forward_training as full
    import inverse_training as inverse
    import basis_sensitivity as prior
    from architecture_controls import hidden_layer
    assert spec in jobs()
    if spec['kind']=='pair':
        full.protocol_test()
        original=study.AdaptiveModel
        class Model(original):
            def __init__(self,case,variant):
                super().__init__(case,spec['input_variant'])
                self.kan[2]=hidden_layer(self.kan[2],spec['hidden_basis'])
                self.variant=variant
        study.AdaptiveModel=Model
        study.gate_diagnostic=lambda model:None
        study.config,study.digest=full.config,full.digest
        module=study
    else:
        module=inverse
        old_data_hashes=inverse.data_hashes
        cache={}
        def arrays(case):
            if case not in cache:
                with np.load(inverse.DATA/(case+'_observations.npz')) as data, np.load(inverse.DATA/(case+'_split.npz')) as split:
                    changed=prior.transform_observations(data['coordinates'],data['target'],data['indices'],case,spec['condition'],spec['job']['seed'])
                    assert not np.intersect1d(changed[2],split['test']).size
                    cache[case]=changed
            return cache[case]
        def load(case,device):
            z,y,_=arrays(case)
            z=torch.tensor(z,device=device,dtype=torch.float32)
            y=torch.tensor(y,device=device,dtype=torch.float32)
            return z,y,y.square().mean().detach()
        def data_hashes(case):
            result=dict(old_data_hashes(case))
            for key,a in zip(('coordinates','observed_target','selected_original_indices'),arrays(case)):
                result['followup_'+key]=hashlib.sha256(a.tobytes(order='C')).hexdigest()
            return result
        inverse.load_observations,inverse.data_hashes=load,data_hashes
    old_cfg,old_digest=module.config,module.digest
    hashes=old_digest()
    extra={p:study.base.sha(Path(__file__).parent/p) for p in
           ('kdv_pairs_and_noise_controls.py','architecture_controls.py','basis_sensitivity.py')}
    def config(case,smoke=False):
        cfg=dict(old_cfg(case,smoke))
        cfg['completion_control']=spec
        if spec['kind']=='robust':
            cfg.update(followup_condition=spec['condition'],noise=.01 if spec['condition']=='noise_1pct' else 0.,
                       observation_fraction=.25 if spec['condition']=='observations_25pct' else 1.,
                       test_split='Frozen prior holdout; removed observations are not test points')
        return cfg
    def digest():
        assert old_digest()==hashes
        assert all(study.base.sha(Path(__file__).parent/n)==v for n,v in extra.items())
        return dict(hashes,**extra)
    module.config,module.digest=config,digest
    return module

def self_test(spec,module,device='cpu'):
    import torch
    import adaptive_basis as study
    import inverse_training as inverse
    study.base.seed(spec['job']['seed'])
    factory=study.AdaptiveModel if spec['kind']=='pair' else inverse.InverseModel
    model=factory(spec['job']['case'],spec['job']['variant']).to(device).double()
    with torch.no_grad():
        for p in model.parameters():p.add_(.001*torch.randn_like(p))
    dim=2 if spec['job']['case']=='KdV' else 3
    z=torch.rand(5,dim,device=device,dtype=torch.float64,requires_grad=True)
    scale=torch.linspace(.3,.9,dim,device=device,dtype=torch.float64)
    q=((z.detach()-.2)/scale).requires_grad_()
    a,b=model(z),model(q*scale+.2)
    errors=[]
    for channel in range(a.shape[1]):
        for axis in range(dim):
            da,db=a[:,channel:channel+1],b[:,channel:channel+1]
            for _ in range(3):
                da=study.base.derivative(da,z)[:,axis:axis+1]
                db=study.base.derivative(db,q)[:,axis:axis+1]/scale[axis]
                assert torch.allclose(da,db,rtol=1e-7,atol=1e-8)
                errors.append(float((da-db).abs().max().detach()))
    (da.square().mean()+a.square().mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    return dict(passed=True,chain_error=max(errors),optimizer_steps=0,spec=spec)
