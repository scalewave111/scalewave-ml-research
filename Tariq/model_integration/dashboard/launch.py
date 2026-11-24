"""Set up one isolated dashboard environment and serve on this PC only."""
from pathlib import Path
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENV = ROOT / '.venv-dashboard'
PYTHON = ENV / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')

def main():
    if not PYTHON.exists():
        subprocess.run([sys.executable, '-m', 'venv', str(ENV)], check=True)
    check = subprocess.run([str(PYTHON), '-c', 'import streamlit, plotly'], capture_output=True)
    if check.returncode:
        lock = HERE / 'requirements.lock.txt'
        requirements = lock if lock.exists() else HERE / 'requirements.txt'
        print('Installing dashboard dependencies once, without download cache...', flush=True)
        subprocess.run([str(PYTHON), '-m', 'pip', 'install', '--no-cache-dir', '-r', str(requirements)], check=True)
    freeze = subprocess.check_output([str(PYTHON), '-m', 'pip', 'freeze'], text=True)
    (HERE / 'requirements.lock.txt').write_text(freeze, encoding='utf-8')
    env = dict(os.environ, STREAMLIT_BROWSER_GATHER_USAGE_STATS='false', PYTHONDONTWRITEBYTECODE='1')
    print('Open http://127.0.0.1:8501 in your browser. Press Ctrl+C here to stop.', flush=True)
    return subprocess.call([str(PYTHON), '-m', 'streamlit', 'run', str(HERE/'app.py'),
                            '--server.address=127.0.0.1', '--server.port=8501', '--server.headless=true',
                            '--browser.gatherUsageStats=false', '--server.fileWatcherType=none'], cwd=HERE, env=env)

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError:
        print('Setup failed. Keep this environment and rerun the same command after resolving the displayed error.', file=sys.stderr)
        sys.exit(1)
