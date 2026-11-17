"""Run all four independent 15m models sequentially; no fits on import."""
from pathlib import Path
import subprocess
import sys

def main():
    runner=Path(__file__).resolve().with_name('fit_net_magnitude_15m.py')
    if any(a.split('=')[0]=='--pure-model' for a in sys.argv[1:]):
        raise SystemExit('Use the individual runner to select one family.')
    for model in ['hgb','extra','mlp','gru']:
        print(f'\nStarting separate 15m {model} run...',flush=True)
        subprocess.run([sys.executable,str(runner),'--pure-model',model,*sys.argv[1:]],check=True)
    print('All four independent 15m runs finished.',flush=True)

if __name__=='__main__':main()
