"""Recompute the archived results using this checkout and an extracted materials bundle."""
import argparse
from uuid import uuid4
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
for stream in (sys.stdout,sys.stderr):
    if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--materials-root',type=Path,default=BASE,
                        help='Extracted bundle directory one level above the folder containing runs.json')
    parser.add_argument('--run-dir',type=Path,help='Output directory (default: a new directory under verification_runs/)')
    parser.add_argument('--resume',action='store_true',help='Continue an existing run in --run-dir')
    args=parser.parse_args()
    materials=args.materials_root.expanduser().resolve()
    archive=materials/'02_training_archive'
    inputs=materials/'03_verification_inputs'
    workbook=materials/'04_verification_results/tables/test_result_comparison.xlsx'
    plan=inputs/'test_plan_and_references'
    missing=[p for p in (archive/'runs.json',inputs/'model_reload_checks.json',
                         plan/'frozen_test_plan.json',plan/'protocol_digest.json',workbook) if not p.is_file()]
    if missing:
        parser.error('Incomplete materials bundle. Check --materials-root. Missing: '+', '.join(map(str,missing)))
    out=(args.run_dir or BASE/'verification_runs'/uuid4().hex[:12]).expanduser().resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        parser.error('The output directory is not empty. Use an empty directory or specify --resume.')
    if args.resume and (out/'execution_state.json').exists():
        previous=read(out/'execution_state.json')
        done={s['name'] for s in previous['steps'] if s['returncode']==0}
        for step,folder,count in (
            ('04_primary_model_tests','independent_tests/predictions',1071),
            ('05_auxiliary_model_tests','auxiliary_test_results',567),
            ('06_wake_model_evaluation','wake_validation_results',114),
        ):
            if step in done and len(list((out/folder).glob('*.npz')))!=count:
                parser.error('Prediction arrays are missing. Start a new run in an empty directory without --resume.')
    out.mkdir(parents=True,exist_ok=True)
    logs=out/'logs';logs.mkdir(exist_ok=True)
    versions={n:importlib.metadata.version(n) for n in ('torch','numpy','scipy','matplotlib','threadpoolctl')}
    environment=dict(python=platform.python_version(),platform=platform.system(),versions=versions,device=args.device)
    (out/'environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2),encoding='utf-8')
    import torch
    if args.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA is unavailable. Use --device cpu.')
    work=out/'independent_tests';work.mkdir(exist_ok=True)
    for name in ('frozen_test_plan.json','protocol_digest.json'):
        source=plan/name
        target=work/name
        if not target.exists():shutil.copy2(source,target)
        assert target.read_bytes()==source.read_bytes(),name
    env={k:v for k,v in os.environ.items() if not k.startswith('ELB_')}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',
               ELB_ARCHIVE_ROOT=str(archive),ELB_DATA_ROOT=str(archive/'training_and_reference_data'),
               ELB_TEST_WORK=str(work),ELB_OUTPUT_ROOT=str(out))
    entry=BASE/'src/cli.py'
    steps=[
        ('01_KANmodel_and_method_checks',['models','--part','models']),
        ('02_reference_convergence',['models','--part','references']),
        ('03_generate_test_references',['test','references']),
        ('04_primary_model_tests',['test','evaluate','--device',args.device]),
        ('05_auxiliary_model_tests',['controls','--device',args.device]),
        ('06_wake_model_evaluation',['wake','--device',args.device]),
        ('07_summarize_tests',['test','summarize']),
        ('08_selection_audit',['selection','--workbook',str(workbook)]),
        ('09_periodicNSdense_boundary_checks',['boundaries','--device',args.device]),
        ('10_recompute_indexed_predictions',['errors',str(inputs),'--run-dir',str(out)]),
    ]
    state_file=out/'execution_state.json'
    state=read(state_file) if args.resume and state_file.exists() else dict(steps=[])
    completed={s['name'] for s in state['steps'] if s['returncode']==0}
    for name,arguments in steps:
        if name in completed:
            print('Resume: already completed',name,flush=True);continue
        print('START',name,flush=True)
        started=time.time()
        command=[sys.executable,'-B','-u',str(entry),*arguments]
        log=logs/(name+'.log')
        with log.open('a',encoding='utf-8') as stream:
            stream.write('\nCOMMAND '+json.dumps([step if not os.path.isabs(step) else Path(step).name for step in command],ensure_ascii=False)+'\n');stream.flush()
            process=subprocess.Popen(command,cwd=BASE,env=env,stdout=stream,stderr=subprocess.STDOUT)
            code=process.wait()
        state['steps'].append(dict(name=name,returncode=code,seconds=round(time.time()-started,3),
                                   log=str(log.relative_to(out))))
        state_file.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
        print('END',name,'exit',code,'seconds',round(time.time()-started,1),flush=True)
        if code:raise SystemExit('Step failed, see log: '+str(log))
        if name=='01_KANmodel_and_method_checks':
            audit=read(out/'method_and_reference_checks/model_checks.json')
            assert not audit['source_hash_failures'] and not audit['failures'], 'Model checks failed'
    # Use recorded errors as comparisons, never as inputs for configuration selection.
    compare=subprocess.run([sys.executable,'-B',str(BASE/'compare_results.py'),str(out)],cwd=BASE,env=env,
                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8')
    (logs/'11_results_comparison.log').write_text(compare.stdout,encoding='utf-8')
    print(compare.stdout,flush=True)
    if compare.returncode:raise SystemExit(compare.returncode)
    state['passed']=True
    state_file.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ALL CHECKS PASSED:',out,flush=True)


if __name__=='__main__':main()
