"""Dense periodic-boundary diagnostics of frozen NS models; no training."""
from pathlib import Path
import sys
import json
import math
import time
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import paths, require, source_is_local, model_directory
CONFIG = paths()
ROOT = CONFIG.archive
OUT = CONFIG.output / "NSboundary_results"
import evaluate_test_set as test
source_is_local(test)


def differences(model, q, axis):
    zero = torch.zeros_like(q[:, :1])
    left = torch.cat((zero, q), 1) if axis == 0 else torch.cat((q[:,:1], zero, q[:,1:]), 1)
    right = left.clone()
    right[:,axis] = 2*math.pi
    left.requires_grad_(); right.requires_grad_()
    results = []
    for z in (left, right):
        u,v,p = test.audit.study.base.ns.velocity_fields(model, z)
        g = model.correction(z)[:,:1]
        du = test.audit.study.base.derivative(u,z)[:,axis:axis+1]
        dv = test.audit.study.base.derivative(v,z)[:,axis:axis+1]
        results.append(torch.cat((g,u,v,p,du,dv),1))
    return results[0]-results[1]


def main(device='cpu'):
    p = test.checked_protocol()
    tasks = [x for x in p['tasks'] if x['eq']=='NS_periodic']
    OUT.mkdir(parents=True, exist_ok=True)
    axes = [np.linspace(0,2*math.pi,257), np.linspace(0,1,65)]
    q = test.mesh(axes)
    protocol = dict(shape=[257,65],points_per_direction=len(q),directions=['x','y'],
        variables=['g','u','v','p','normal_derivative_u','normal_derivative_v'],
        dtype='float64',models=102,scope='Main periodic NS: 16 pairs + MLP, forward/inverse, seeds 42,43,44',
        statistic='RMS and maximum absolute mismatch across both paired boundaries; raw nondimensional variables',
        limitation='Finite dense grids measure boundary mismatch, not a proof of an everywhere bound',
        parent_protocol_sha256=test.sha(test.PROTOCOL))
    pp=OUT/'plan.json'
    if not pp.exists():
        test.write_new(pp,protocol)
        np.savez_compressed(OUT/'boundary_coordinates.npz',tangential_time=q)
    else:assert test.read(pp)==protocol
    rows=[]; started=time.perf_counter()
    for task in tasks:
        for r in task['runs']:
            dest=OUT/(r['id']+'.json')
            if dest.exists():rows.append(test.read(dest));continue
            model,_=test.audit.make_model(r)
            weight=model_directory(r['run'])/'final.pt'
            assert test.sha(weight)==r['weights_sha256']
            model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
            model=model.double().to(device).eval()
            values=[]
            for axis in (0,1):
                chunks=[]
                for start in range(0,len(q),256):
                    z=torch.tensor(q[start:start+256],device=device)
                    value=differences(model,z,axis)
                    chunks.append(value.detach().cpu().numpy())
                    del value,z
                values.append(np.concatenate(chunks))
            delta=np.stack(values)
            assert np.isfinite(delta).all()
            coarse=delta.reshape(2,257,65,6)[:,::2,::2,:].reshape(-1,6)
            flat=delta.reshape(-1,6)
            probe=torch.tensor(q[np.linspace(0,len(q)-1,31,dtype=int)],device=device)
            manual=sum(differences(model,probe,a).square().mean(0).sum() for a in (0,1))
            original=test.audit.study.base.boundary(model,probe)
            formula_error=float(abs(manual-original).detach().cpu())
            assert formula_error<1e-10
            npz=OUT/(r['id']+'.npz')
            np.savez_compressed(npz,boundary_difference=delta)
            row=dict(id=r['id'],mode=r['mode'],model=r['model'],seed=r['seed'],
                rms=np.sqrt(np.mean(flat**2,axis=0)).tolist(),maximum=np.max(np.abs(flat),axis=0).tolist(),
                coarse_rms=np.sqrt(np.mean(coarse**2,axis=0)).tolist(),coarse_maximum=np.max(np.abs(coarse),axis=0).tolist(),
                loss_fine=float(2*np.mean(flat**2,axis=0).sum()),
                formula_max_difference=formula_error,weights_sha256=r['weights_sha256'],
                mismatch_file_sha256=test.sha(npz),coordinates_sha256=test.sha(OUT/'boundary_coordinates.npz'))
            test.write_new(dest,row); rows.append(row)
            del model
            if len(rows)%6==0:print('NS boundary',len(rows),'/102',round(time.perf_counter()-started,1),flush=True)
    target=OUT/'boundary_checks.json'
    if not target.exists():test.write_new(target,dict(protocol=protocol,runs=rows))
    for task in tasks:
        for name in dict.fromkeys(('MLP',task['same_family'],task['selected'])):
            group=[r for r in rows if r['mode']==task['mode'] and r['model']==name]
            print(task['id'],name,'mean RMS',np.mean([r['rms'] for r in group],axis=0).tolist(),
                  'worst maxima',np.max([r['maximum'] for r in group],axis=0).tolist(),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=parser.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Use --device cpu.')
    require(test.PROTOCOL)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1): main(args.device)
