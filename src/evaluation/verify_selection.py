"""Check experimental test comparisons against choices frozen on validation data."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import math
import statistics
import sys
import zipfile
import xml.etree.ElementTree as ET
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment_paths import paths, require, source_is_local
CONFIG=paths()
ROOT=CONFIG.archive
import evaluate_test_set as test
source_is_local(test)


def main(workbook=None):
    p=test.checked_protocol()
    original=CONFIG.runs
    assert test.sha(original)==p['source_validation_records_sha256']
    old=test.read(original)
    summary=test.read(test.OUT/'summary_data.json')
    assert summary['selected_using_new_test'] is False
    assert summary['protocol_sha256']==test.sha(test.PROTOCOL)
    assert p['selection_locked_before_new_predictions'] is True
    assert p['new_test_used_for_selection'] is False
    tests={x['task']:x for x in summary['summary']}
    decisions=[]
    for task in p['tasks']:
        rows=[r for r in old if (r['eq'],r['mode'],r['reg'])==(task['eq'],task['mode'],task['reg'])]
        validation={}
        for name in test.PAIRS:
            selected=sorted([r for r in rows if r['model']==name],key=lambda x:x['seed'])
            assert [r['seed'] for r in selected]==[42,43,44]
            validation[name]=statistics.mean(r['Eu'] for r in selected)
            assert validation[name]==task['validation_means'][name]
        selected=min(validation,key=validation.get)
        homogeneous=min((n for n,(a,b) in test.PAIRS.items() if a==b),key=validation.get)
        assert selected==task['selected']==tests[task['id']]['selected']
        assert homogeneous==task['same_family']==tests[task['id']]['same']
        mean={}
        for name in dict.fromkeys((selected,homogeneous,'MLP')):
            members=[r for r in task['runs'] if r['model']==name]
            records=[test.read(test.RECORDS/(r['id']+'.json')) for r in members]
            assert len(records)==3
            assert all(r['protocol_sha256']==test.sha(test.PROTOCOL) for r in records)
            mean[name]=statistics.mean(r['new_test_error'] for r in records)
        r=tests[task['id']]
        for actual,expected in [(r['new_selected'],mean[selected]),(r['new_same'],mean[homogeneous]),
            (r['new_mlp'],mean['MLP']),(r['new_gain_mlp'],mean['MLP']/mean[selected]),
            (r['new_reduction_same'],100*(1-mean[selected]/mean[homogeneous]))]:
            assert math.isclose(actual,expected,rel_tol=1e-13,abs_tol=1e-13)
        decisions.append(dict(task=task['id'],selected=selected,same=homogeneous,
            validation_mean_selected=validation[selected],test_mean_selected=mean[selected]))
    records=list(test.RECORDS.glob('*.json'))
    assert len(records)==1071
    for path in records:assert test.read(path)['protocol_sha256']==test.sha(test.PROTOCOL)
    assert test.PROTOCOL.stat().st_mtime < min(x.stat().st_mtime for x in records)
    workbook=workbook or CONFIG.workbook
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    formulas=[]
    with zipfile.ZipFile(workbook) as z:
        for name in z.namelist():
            if name.startswith('xl/worksheets/sheet') and name.endswith('.xml'):
                root=ET.fromstring(z.read(name))
                formulas.extend(n.text or '' for n in root.findall('.//s:f',ns))
    assert not any(token in f.upper() for f in formulas for token in ('MIN(', 'SMALL(', 'RANK('))
    result=dict(passed=True,tasks=21,models=1071,validation_choices_match=True,
        test_results_use_frozen_choices=True,test_based_reselection=False,
        diagnostic_ranking='Post-evaluation rankings did not change the selected configurations, checkpoints, or reported comparisons.',
        protocol_sha256=test.sha(test.PROTOCOL),validation_source_sha256=test.sha(original),
        workbook_sha256=test.sha(workbook),workbook_has_test_selection_formula=False,
        decisions=decisions)
    CONFIG.output.mkdir(parents=True,exist_ok=True)
    target=CONFIG.output/'selection_audit.json'
    with target.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='decisions'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook',type=Path,
                        default=CONFIG.workbook,
                        help='comparison workbook, may be outside the source tree')
    args=parser.parse_args()
    require(CONFIG.runs, test.PROTOCOL, test.OUT/'summary_data.json',
            args.workbook)
    main(args.workbook)
