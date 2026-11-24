"""Publish verified comparison and apply the MAE-only benchmark rule."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'run_001'
config=json.loads((OUT/'config.json').read_text())
verification=json.loads((OUT/'verification.json').read_text())
assert verification['status']=='PASS'
result=json.loads((OUT/'RESULT.json').read_text())
pred=pd.read_parquet(OUT/'predictions.parquet')
baseline=pd.read_parquet(ROOT/config['benchmark_predictions'])
registry_path=HERE.parent/'magnitude_benchmark_registry.json'
registry=json.loads(registry_path.read_text())
assert registry['metric_policy']['primary_metrics']=={'MAE':'lower'}
assert abs(registry['current']['MAE']-config['benchmark_mae'])<1e-12
assert not any(e['experiment']==config['version'] for e in registry['evaluated_experiments'])
names=['incumbent_hybrid']+list(config['additions'])
rows=[]
for name in names:
    rows.append({'model':name,'MAE':result['pooled_MAE'][name],
        'MAE_delta_vs_incumbent':result['pooled_MAE'][name]-config['benchmark_mae'],
        'Spearman_IC':result['pooled_Spearman_IC'][name],
        'Pearson_IC':result['pooled_Pearson_IC'][name]})
pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
diagnostics=[]
for group_name,groups in [('fold',baseline.fold),('month',pd.Series(pred.index.strftime('%Y-%m'),index=pred.index))]:
    for key,frame in pred.groupby(groups):
        for name in names:
            diagnostics.append({'group':group_name,'period':key,'model':name,'rows':len(frame),
                'MAE':float(np.abs(frame.actual-frame[name]).mean()),
                'Spearman_IC':float(spearmanr(frame.actual,frame[name]).statistic)})
pd.DataFrame(diagnostics).to_csv(OUT/'period_metrics.csv',index=False)
winner=min(names,key=lambda n:result['pooled_MAE'][n])
promote=result['pooled_MAE'][winner]<registry['current']['MAE']-registry['metric_policy']['numerical_tolerance']
decision={'decision':'PROMOTE' if promote else 'NO_PROMOTION','lowest_MAE_model':winner,
    'criterion':'Verified same-row MAE only; Spearman tracked without a promotion gate.',
    'previous_benchmark':registry['current'].copy(),
    'candidate_scores':{n:result['pooled_MAE'][n] for n in config['additions']},
    'verification':str((OUT/'verification.json').relative_to(ROOT))}
if promote:
    registry['current']={'experiment':config['version'],
        'prediction_file':str((OUT/'predictions.parquet').relative_to(ROOT)),
        'prediction_column':winner,'MAE':result['pooled_MAE'][winner],
        'Spearman_IC':result['pooled_Spearman_IC'][winner],
        'Pearson_IC':result['pooled_Pearson_IC'][winner],
        'verification':decision['verification'],'IC_verification':decision['verification'],
        'prediction_sha256':hashlib.sha256((OUT/'predictions.parquet').read_bytes()).hexdigest()}
    registry['history'].append({'model':config['version']+'/'+winner,
        'MAE':registry['current']['MAE'],'Spearman_IC':registry['current']['Spearman_IC'],
        'Pearson_IC':registry['current']['Pearson_IC']})
decision['current_benchmark']=registry['current'].copy()
with (OUT/'decision.json').open('x') as f:
    json.dump(decision,f,indent=2)
registry['evaluated_experiments'].append({'experiment':config['version'],
    'decision':decision['decision'],'candidate_scores':decision['candidate_scores'],
    'verification':decision['verification'],'new_fits_including_weight_selection':config['new_fit_budget']})
registry_path.write_text(json.dumps(registry,indent=2)+'\n')
lines=['MEDIAN-REGRESSION FEATURE ADDITIONS: VERIFIED RESULTS',
    'Same 48-hour target, same 18,481 scoring rows, same four quarterly outer windows.',
    'One feature added to the liquidation median regression in each experiment.',
    'RF models unchanged. Blend weights selected by MAE using earlier three-month inner validation.',
    'Strict label-end purging and training-only preprocessing; fresh test not accessed.',
    '',pd.DataFrame(rows).to_string(index=False),'',
    'Selected RF weights by fold: '+json.dumps(result['selected_forest_weights']),
    'Decision: '+decision['decision']+'; benchmark MAE '+str(registry['current']['MAE']),
    '24 model fits plus 12 historical blend selections. No additional tuning or combinations.',
    'Spearman is diagnostic only. Nominal block intervals do not account for adaptive experiment selection.',
    'See RESULT.json for paired 7-/14-day block intervals and monthly improvement counts.',
    'Other horizons require matching labels, cached RF models, fold dates and separate benchmarks.']
with (OUT/'SUMMARY.txt').open('x') as f:
    f.write('\n'.join(lines)+'\n')
print(pd.DataFrame(rows).to_string(index=False))
print(json.dumps(decision,indent=2))
