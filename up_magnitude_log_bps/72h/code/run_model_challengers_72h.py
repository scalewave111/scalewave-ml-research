"""Run the Extra Trees and neural-network 72h experiments sequentially."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import subprocess
import sys
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    here = Path(__file__).resolve().parent
    root = here/'experiments/model_challengers_72h_v1'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if not args.check_inputs:
        root.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, filename in [('extra_trees', 'fit_extra_trees_excursions_72h.py'),
                           ('neural', 'fit_neural_excursions_72h.py')]:
        command = [sys.executable, str(here/filename), '--threads', str(args.threads)]
        command += ['--check-inputs'] if args.check_inputs else ['--output-dir', str(root/name)]
        subprocess.run(command, check=True)
        if not args.check_inputs:
            paths = list((root/name).glob('*/COMPLETE.json'))
            if len(paths) != 1 or json.loads(paths[0].read_text())['status'] != 'COMPLETE':
                raise RuntimeError('Missing completed result for '+name)
            results[name] = str(paths[0].parent)
    if not args.check_inputs:
        (root/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE', runs=results), indent=2), encoding='utf-8')
        print('Both model experiments completed:', root, flush=True)


if __name__ == '__main__':
    main()
