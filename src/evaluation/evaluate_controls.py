"""Evaluate all remaining non-wake archived models on the fixed new grids."""
import argparse
import ast
import json
import math
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment_paths import paths, require, source_is_local, model_directory
CONFIG=paths()
ROOT=CONFIG.archive
OUT=CONFIG.output/'auxiliary_test_results'
import evaluate_test_set as test
source_is_local(test)
audit=test.audit


def make_model(r):
    name,eq=r['model'],r['eq']
    if eq=='NS_wake':
        audit.wake.DATA=CONFIG.data/'wake_observations'
        audit.inverse.INITIAL[eq]=[1.5,.02]
    if name in audit.PAIRS or name=='MLP':
        model,net=audit.make_model(r)
        if r['reg']=='fourier_4terms':
            net.kan[0]=audit.study.VariableBasisLayer(2,32,4,'Fourier')
        return model
    torch.manual_seed(r['seed'])
    inv=r['mode']=='inverse'
    if eq in audit.extended.CASES:
        spec=dict(case=eq,regime=r['reg'],variant=name)
        return audit.ext_inverse.Model(spec) if inv else audit.extended.Model(spec)
    factory=audit.burgers.BurgersModel if eq=='Burgers' else (
        audit.wake.Network if eq=='NS_wake' else audit.ORIGINAL_FACTORY)
    if 'architecture_control' in r['cfg']:
        src=audit.CODE/'architecture_controls.py'
        tree=ast.parse(src.read_text(encoding='utf-8-sig'))
        install=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install')
        cls=next(n for n in install.body if isinstance(n,ast.ClassDef) and n.name=='ControlModel')
        scope=dict(original_model=factory,spec=r['cfg']['architecture_control'],nn=nn,torch=torch,
            math=math,study=audit.study,hidden_layer=audit.hidden_layer,case=eq)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(src),'exec'),scope)
        factory=scope['ControlModel']
    elif name=='SharedLearned' and eq!='Burgers':
        original=factory
        def factory(case,variant):
            net=original(case,'Adaptive')
            net.kan[0].logits=nn.Parameter(net.kan[0].logits[:1].detach().clone())
            net.variant=variant
            return net
    if not inv:return factory(eq,name)
    audit.study.AdaptiveModel=factory
    try:return audit.inverse.InverseModel(eq,name)
    finally:audit.study.AdaptiveModel=audit.ORIGINAL_FACTORY


def main(device='cpu'):
    main_protocol=test.checked_protocol()
    all_rows=test.read(CONFIG.runs)
    completed={r['run'] for t in main_protocol['tasks'] for r in t['runs']}
    rows=[r for r in all_rows if r['eq']!='NS_wake' and r['run'] not in completed]
    assert len(rows)==567
    OUT.mkdir(parents=True,exist_ok=True)
    freeze=OUT/'frozen_plan.json'
    if not freeze.exists():
        locked=[]
        for r in rows:
            source=model_directory(r['run']); meta=test.read(source/'result.json')
            assert test.sha(source/'final.pt')==meta['weights_sha256']
            locked.append(dict(r,weights_sha256=meta['weights_sha256']))
        test.write_new(freeze,dict(parent_protocol_sha256=test.sha(test.PROTOCOL),models=locked,
            scope='567 auxiliary models, excluding 114 wake validation runs',
            selection='No configuration re-selection; grids and reference definitions inherited without modification'))
    frozen=test.read(freeze)
    assert frozen['parent_protocol_sha256']==test.sha(test.PROTOCOL)
    records=[];refs={};started=time.perf_counter()
    for number,r in enumerate(frozen['models'],1):
        key=test.task_id(r['eq'],r['mode'],r['reg'])+'_'+r['model']+'_seed'+str(r['seed'])
        dest=OUT/(key+'.json')
        if dest.exists():records.append(test.read(dest));continue
        rid=test.ref_id(r['eq'],r['reg'])
        if rid not in refs:
            path=test.REF/(rid+'.npz');meta=test.read(test.REF/(rid+'.json'))
            assert test.sha(path)==meta['file_sha256']
            refs[rid]=(audit.loadz(path),meta)
        ref,meta=refs[rid]
        weight=model_directory(r['run'])/'final.pt'
        assert test.sha(weight)==r['weights_sha256']
        model=make_model(r)
        assert sum(x.numel() for x in model.parameters())==r['n'],key
        model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
        model=model.double().to(device).eval()
        oldz,oldpred=audit.load_reload_probe(r)
        probe=test.prediction(model,torch.tensor(oldz,device=device),r['eq'])
        delta=float(np.max(np.abs(probe.detach().cpu().numpy()-oldpred)))
        assert delta<1e-8,(key,delta)
        chunks=[]
        for start in range(0,len(ref['coordinates']),512):
            z=torch.tensor(ref['coordinates'][start:start+512],device=device)
            out=test.prediction(model,z,r['eq'])
            chunks.append(out.detach().cpu().numpy())
            del out,z
        values=np.concatenate(chunks); exact=ref['reference']
        eu=float(100*np.linalg.norm(values-exact)/np.linalg.norm(exact))
        independent=float(100*np.sqrt(np.sum((values-exact)**2)/np.sum(exact**2)))
        assert np.isclose(eu,independent,atol=1e-12,rtol=1e-12)
        npz=OUT/(key+'.npz')
        np.savez_compressed(npz,prediction=values,test_index=ref['test_index'],reference_id=np.array(rid))
        record=dict(id=key,eq=r['eq'],mode=r['mode'],reg=r['reg'],model=r['model'],seed=r['seed'],
            run=r['run'],old_error=r['Eu'],new_error=eu,points=len(values),reload_max_difference=delta,
            max_absolute_error=float(np.max(np.abs(values-exact))),weights_sha256=r['weights_sha256'],
            prediction_sha256=test.sha(npz),reference_sha256=meta['file_sha256'],parameters=r['n'])
        test.write_new(dest,record);records.append(record)
        del model,probe,values,chunks
        if number%20==0:print('Auxiliary',number,'/567',round(time.perf_counter()-started,1),flush=True)
    target=OUT/'auxiliary_results.json'
    if not target.exists():test.write_new(target,dict(runs=records,protocol_sha256=test.sha(freeze)))
    print('Completed auxiliary models:',len(records),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=parser.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Use --device cpu.')
    require(CONFIG.runs, test.PROTOCOL, test.REF)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main(args.device)
