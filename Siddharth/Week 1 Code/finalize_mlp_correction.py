from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/mlp_baseline_correction_v5'
r=json.loads((OUT/'COMPLETE.json').read_text())
assert r['status']=='EXECUTED_AND_AUDITED' and r['audit']['status']=='PASS'
ds=json.loads((OUT/'SELECTIONS.json').read_text());rows=[]
for d in ds:
 s=d['selected'];c=s.get('config') or {}
 rows.append(dict(issue=d['issue'],model=s['id'],strength=s['strength'],widths=c.get('widths',[]),learning_rate=c.get('lr'),validation_gain_bps=s['gain_bps'],eligible_candidates=d['eligible_candidates'],outer_epochs=d['outer_epochs'],monthly_gains=s.get('monthly_gains',[]),seed_gains=s.get('seed_gains',[])))
selection=pd.DataFrame(rows);selection.to_csv(OUT/'selected_corrections.csv',index=False)
comparison=pd.read_csv(ROOT/'experiments/mlp_hyperparameter_tuning_v4/comparison.csv');m=r['test_scores']['corrected']
comparison=pd.concat([comparison,pd.DataFrame([dict(model='Baseline + guarded MLP correction',mae_bps=m['mae'],spearman_ic=m['rank_ic'],pearson_ic=m['pearson_ic'],mae_difference_vs_baseline=r['mae_difference_corrected_minus_baseline'])])],ignore_index=True)
comparison.to_csv(OUT/'comparison.csv',index=False)
text='Baseline plus a small two-input MLP correction: executed on real data and audited.\n\nThe MLP uses only the original 5m and 15m liquidation historical active rates. Correction targets are actual outcomes minus chronological held-out baseline predictions, never baseline training residuals. Baseline OOF predictions start January 2023 after warmup, and reproduce the saved 2024-2025 baseline. The baseline retains its original two-source-feature recipe and missing-value handling; the correction network has exactly two inputs.\n\n'
text+='Six tiny configurations, three seeds, and three preceding monthly validation folds. Strengths: 0, .05, .1, .25, .5, 1. A nonzero correction requires pooled validation gain >=0.1 bps, positive gains in at least two months and two seeds, and worst monthly degradation <=1 bps. Raw correction is capped at +/-100 bps. Zero correction is the fallback. Same four outer walk-forward windows and 18,481 evaluation rows.\n\n'
text+=comparison.to_string(index=False)+'\n\nSelected corrections:\n'+selection.to_string(index=False)
text+='\n\nPer-window results:\n'+pd.read_csv(OUT/'window_scores.csv').to_string(index=False)
text+='\n\nAudit and uncertainty:\n'+json.dumps(r,indent=2)
text+='\n\nThe historical evaluation was previously examined. This is an exploratory comparison, not untouched holdout evidence. The experiment was frozen before its outer results were scored.\n'
(OUT/'RESULTS.txt').write_text(text,encoding='utf-8')
nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},cells=[
 dict(id='correction-intro',cell_type='markdown',metadata={},source=['# Two-feature MLP baseline correction\n','The correction learns chronological held-out baseline errors from only the original 5m and 15m liquidation features. Validation can choose zero correction. The historical evaluation has already been examined.\n']),
 dict(id='correction-results',cell_type='code',metadata={},execution_count=None,outputs=[],source=["from pathlib import Path\n","import pandas as pd, json\n","run=Path('experiments/mlp_baseline_correction_v5')\n","display(pd.read_csv(run/'scores.csv'))\n","display(pd.read_csv(run/'selected_corrections.csv'))\n","display(pd.read_csv(run/'window_scores.csv'))\n","display(pd.read_csv(run/'comparison.csv'))\n","print((run/'COMPLETE.json').read_text())\n"]),
 dict(id='correction-provenance',cell_type='markdown',metadata={},source=['Runner: mlp_baseline_correction.py. The run folder contains baseline OOF predictions and provenance, original baseline model copies/new historical refits, correction checkpoints, validation trials, selected strengths, predictions, source/configuration, and audit. This notebook displays saved results without starting training.'])])
for c in nb['cells']:
 if c['cell_type']=='code':compile(''.join(c['source']),'notebook','exec')
(ROOT/'BTC_48h_MLP_Baseline_Correction.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
context=ROOT/'PROJECT_CONTEXT.md';old=context.read_text(encoding='utf-8');marker='## Guarded MLP baseline correction result — 2026-11-04'
if marker not in old:
 b=r['test_scores']['baseline']
 note='\n\n'+marker+'\n\nUser authorized the proposed baseline-error correction using only the original two liquidation inputs. Executed on real data and audited. Baseline chronological OOF predictions start January 2023, using older data for baseline warmup; correction training never uses in-sample baseline residuals. The baseline itself retains all original expanding training data. Three earlier monthly folds and three seeds select among six tiny correction networks and strengths including zero. Nonzero selection requires pooled gain >=0.1 bps, gains in >=2 months and >=2 seeds, and worst monthly degradation <=1 bps. Outer windows and 18,481 evaluation rows are unchanged.\n\n'
 note+=f"- Baseline MAE {b['mae']:.6f} bps; Spearman IC {b['rank_ic']:.6f}.\n- Corrected MAE {m['mae']:.6f} bps; Spearman IC {m['rank_ic']:.6f}.\n- Beats baseline MAE: {r['beats_baseline']}; active correction windows: {r['active_correction_windows']}/4.\n"
 note+='\nSources: `experiments/mlp_baseline_correction_v5/COMPLETE.json`, `AUDIT.json`, `RESULTS.txt`, `comparison.csv`, `selected_corrections.csv`, `BASELINE_SELECTIONS.json`, `baseline_oof_predictions.csv`, and `window_scores.csv`. Runner: `mlp_baseline_correction.py`. Results viewer: `BTC_48h_MLP_Baseline_Correction.ipynb`. Baseline OOF replay, original baseline reconstruction, residual provenance, label purges, preprocessing, eligibility selection and correction replay checks passed. Prior results and raw data remain unchanged. Historical evaluation remains exploratory, not untouched holdout evidence.\n'
 context.write_text(old+note,encoding='utf-8')
print(pd.read_csv(OUT/'scores.csv').to_string(index=False))
print(selection.to_string(index=False))
print('Saved report, checked notebook code, and updated project context.')
