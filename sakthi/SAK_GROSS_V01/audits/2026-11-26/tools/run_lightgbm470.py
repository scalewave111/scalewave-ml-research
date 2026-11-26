"""Run supplied code with an isolated LightGBM 4.7 wheel; global packages unchanged."""
import os,sys,runpy
from pathlib import Path
sys.dont_write_bytecode=True
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='2'
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'audit_lightgbm470'))
import lightgbm
assert lightgbm.__version__=='4.7.0',lightgbm.__version__
script=Path(sys.argv[1]).resolve()
sys.argv=sys.argv[1:]
sys.path.insert(0,str(script.parent))
print('CONTROLLED_LIGHTGBM_VERSION',lightgbm.__version__,flush=True)
runpy.run_path(str(script),run_name='__main__')
