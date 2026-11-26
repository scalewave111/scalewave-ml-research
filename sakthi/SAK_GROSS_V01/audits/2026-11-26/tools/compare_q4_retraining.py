from pathlib import Path
import json,argparse
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'source/SAK_GROSS_V01_COMPLETE'
ap=argparse.ArgumentParser()
ap.add_argument('--replay',default='retrained_q4_replay')
ap.add_argument('--weights',default='retrained_q4')
ap.add_argument('--output',default='Q4_RETRAINING_AUDIT.json')
ap.add_argument('--lightgbm-version',default='4.6.0')
args=ap.parse_args()
before=pd.read_parquet(SOURCE/'evidence/replay/2024Q4.parquet')
after=pd.read_parquet(ROOT/args.replay/'2024Q4.parquet')
assert before.index.equals(after.index)
errors={c:float(np.max(np.abs(before[c].to_numpy()-after[c].to_numpy()))) for c in ['p_up','predicted_magnitude_bps','signed_score_bps','actual_bps','gross_bps']}
changed={c:int((before[c].to_numpy()!=after[c].to_numpy()).sum()) for c in ['eligible','predicted_side','trade_side']}
name='SAK_DIR_V01_2024Q4_calibration.json'
cal_before=json.loads((SOURCE/'model_files'/name).read_text());cal_after=json.loads((ROOT/args.weights/name).read_text())
exact=all(x==0 for x in errors.values()) and all(x==0 for x in changed.values()) and cal_before==cal_after
within_tolerance=max(errors.values())<1e-10 and all(x==0 for x in changed.values()) and cal_before==cal_after
result={'status':'EXACT' if exact else 'PASS_NUMERICAL_TOLERANCE' if within_tolerance else 'MISMATCH','numerical_tolerance':1e-10,'scope':'Independent latest-quarter retraining with unchanged supplied settings and independently reconstructed H4 inputs','quarter':'2024Q4','rows':len(after),'lightgbm_version':args.lightgbm_version,'max_absolute_errors':errors,'changed_decisions':changed,'calibration_exact':cal_before==cal_after,'metrics':json.loads((ROOT/args.replay/'metrics.json').read_text()),'runtime_note':f'Audit Python 3.12.8 / LightGBM {args.lightgbm_version} / SciPy 1.17.0. Supplied environment Python 3.13.9 / LightGBM 4.7.0 / SciPy 1.16.3. No global dependencies were changed.','not_repeated':'Retraining all seven quarters in the submitted environment remains a provided-package claim, not a new audit result.'}
(ROOT/args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
