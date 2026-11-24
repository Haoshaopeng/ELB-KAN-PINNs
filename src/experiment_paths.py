"""Locate the training archive, reference data and evaluation outputs."""
import argparse
from functools import lru_cache
import os
from pathlib import Path
import sys
from types import SimpleNamespace

PACKAGE = Path(__file__).resolve().parent


def path_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--archive-root', type=Path, help='Directory containing runs.json and archived models')
    parser.add_argument('--data-root', type=Path, help='Training and reference data directory')
    parser.add_argument('--test-work', type=Path, help='Directory containing the frozen test protocol and test references')
    parser.add_argument('--output-root', type=Path, help='Destination for verification reports')
    return parser


def _path(value, variable, default):
    return Path(value or os.environ.get(variable) or default).expanduser().resolve()


@lru_cache(maxsize=1)
def paths():
    args, remaining = path_parser().parse_known_args()
    sys.argv[1:] = remaining
    archive = _path(args.archive_root, 'ELB_ARCHIVE_ROOT', PACKAGE.parent / '02_training_archive')
    data = _path(args.data_root, 'ELB_DATA_ROOT', archive / 'training_and_reference_data')
    test = _path(args.test_work, 'ELB_TEST_WORK', archive.parent / '03_verification_inputs/test_plan_and_references')
    output = _path(args.output_root, 'ELB_OUTPUT_ROOT', PACKAGE.parent / '04_verification_results')
    # Keep source imports local; archives supply data and weights, never Python modules.
    code = PACKAGE / 'training'
    evaluation = PACKAGE / 'evaluation'
    for folder in (evaluation, code):
        if str(folder) in sys.path:
            sys.path.remove(str(folder))
        sys.path.insert(0, str(folder))
    sys.dont_write_bytecode = True
    return SimpleNamespace(package=PACKAGE, archive=archive, data=data, test=test,
                           output=output, code=code,
                           reference=PACKAGE / 'references',
                           runs=archive / 'runs.json',
                           workbook=archive.parent / '04_verification_results/tables/test_result_comparison.xlsx')


def model_directory(run):
    """Resolve model locations while retaining frozen experiment records."""
    archive = paths().archive
    relative = Path(run)
    recorded = archive / relative
    if recorded.exists():
        return recorded
    if relative.parts and relative.parts[0] == '05_experiment_results':
        return archive / 'models_and_records' / Path(*relative.parts[1:])
    return recorded


def require(*items):
    missing = [str(Path(item)) for item in items if not Path(item).exists()]
    if missing:
        raise FileNotFoundError('Missing archive inputs:\n' + '\n'.join(missing) +
                                '\nSet --archive-root, --data-root or --test-work to the matching archive.')


def source_is_local(module):
    path = Path(module.__file__).resolve()
    if PACKAGE not in path.parents:
        raise RuntimeError('Loaded a module outside the source tree: ' + str(path))

