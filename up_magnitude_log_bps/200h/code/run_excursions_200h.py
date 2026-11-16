"""Run the 200-hour excursion search, then its price-feature follow-up."""
import argparse
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-inputs', action='store_true')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    search = [sys.executable, str(here/'discover_excursion_features_200h.py')]
    if args.check_inputs:
        subprocess.run(search + ['--check-inputs'], check=True)
        return
    root = here/'experiments/excursion_pipeline_200h_v1'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root.mkdir(parents=True, exist_ok=False)
    subprocess.run(search + ['--output-dir', str(root/'search')], check=True)
    completed = list((root/'search').glob('*/COMPLETE.json'))
    if len(completed) != 1:
        raise RuntimeError('Expected exactly one completed first-stage run')
    subprocess.run([sys.executable, str(here/'improve_excursion_features_200h.py'),
                    '--reference-run', str(completed[0].parent),
                    '--output-dir', str(root/'price_followup')], check=True)
    print('Both 200-hour stages completed:', root, flush=True)


if __name__ == '__main__':
    main()
