"""Evaluate archived models and reference solutions.

Examples (from the repository root):
  python src/cli.py paths
  python src/cli.py models --archive-root ARCHIVE --part models
  python src/cli.py test --test-work TEST_WORK --help
"""
import argparse
import json
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
from experiment_paths import path_parser, paths

SCRIPTS = {
    'errors': 'recompute_errors.py',
    'models': 'verify_models.py',
    'test': 'evaluate_test_set.py',
    'controls': 'evaluate_controls.py',
    'wake': 'evaluate_wake.py',
    'boundaries': 'periodic_boundaries.py',
    'selection': 'verify_selection.py',
    'kdv-reference': 'verify_models.py',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, parents=[path_parser()],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['paths', *SCRIPTS])
    # Resolve shared flags before the selected script parses its own arguments.
    cfg = paths()
    args, remaining = parser.parse_known_args()
    if args.action == 'paths':
        print(json.dumps({k: str(v) for k, v in vars(cfg).items()}, ensure_ascii=False, indent=2))
        return
    if args.action == 'kdv-reference':
        if remaining:
            parser.error('kdv-reference accepts only the shared path options; use models for other checks.')
        remaining = ['--part', 'kdv']
    script = cfg.package / 'evaluation' / SCRIPTS[args.action]
    sys.argv = [str(script), *remaining]
    try:
        runpy.run_path(str(script), run_name='__main__')
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == '__main__':
    main()
