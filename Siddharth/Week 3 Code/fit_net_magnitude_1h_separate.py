"""Run HGB, ExtraTrees, MLP and GRU separately across every 1h evaluation period.

python Code/fit_net_magnitude_1h_separate.py
All extra arguments are forwarded to the individual runner. No fits on import.
"""
from pathlib import Path
import subprocess
import sys


def main():
    runner=Path(__file__).resolve().with_name('fit_net_magnitude_1h.py')
    forwarded=sys.argv[1:]
    forbidden=['--pure-model','--hgb-only','--models']
    if any(arg.split('=')[0] in forbidden for arg in forwarded):
        raise SystemExit('This runner fixes four independent model families; use the individual runner to choose one.')
    for model in ['hgb','extra','mlp','gru']:
        print(f'\nStarting independent {model} run across all periods...',flush=True)
        subprocess.run([sys.executable,str(runner),'--pure-model',model,*forwarded],check=True)
    print('\nAll four independent runs finished. Compare each report model row on the same evaluation timestamps.',flush=True)


if __name__=='__main__':
    main()
