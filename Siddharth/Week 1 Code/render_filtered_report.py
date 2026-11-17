"""Render filtered study report from saved evidence; no training."""
from pathlib import Path
import sys,json
import pandas as pd
out=Path(sys.argv[1]);c=json.loads((out/'CONTRACT.json').read_text());r=json.loads((out/'RESULT.json').read_text())
scores=pd.read_csv(out/'scores.csv');test=scores.query("split=='test'").set_index('model')
monthly=pd.read_csv(out/'monthly_scores.csv');contrib=pd.read_csv(out/'validation_contributions.csv')
choices=[json.loads(p.read_text()) for p in sorted((out/'outer').glob('*_selection.json'))]
changes=[]
for before,after,label in [('cat_core_exp','cat_robust_exp','Robust feature transformation'),('cat_robust_exp','cat_engineered_exp','Causal engineering'),('cat_robust_exp','cat_state_exp','Matured target history'),('cat_state_exp','cat_normalized_exp','Target scale normalization')]:
 a=test.loc[before];b=test.loc[after]
 changes.append(dict(change=label,before_mae=a.mae,after_mae=b.mae,mae_improvement=a.mae-b.mae,
  before_ic=a.rank_ic,after_ic=b.rank_ic,ic_improvement=b.rank_ic-a.rank_ic))
pd.DataFrame(changes).to_csv(out/'engineering_comparison.csv',index=False)
def table(f):return f.to_html(index=False,border=0,float_format=lambda v:f'{v:.6f}',na_rep='undefined')
admission=[]
for p in sorted((out/'filters').glob('*__exp.csv')):
 f=pd.read_csv(p)
 for fam,g in f.groupby('family'):
  admission.append(dict(issuance=p.name[:7],family=fam,screened=len(g),passed=int(g.passed.sum())))
selection=pd.DataFrame([dict(outer=s['outer'],recipe=s['selected']['recipe'],validation_rows=s['inner_rows']) for s in choices])
page=f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Filtered walk-forward research results</title>
<style>body{{font:16px/1.5 Segoe UI,sans-serif;max-width:1160px;margin:40px auto;padding:0 24px;color:#192b39;background:#fafcfd}}table{{border-collapse:collapse;width:100%;background:white;margin:18px 0 32px;font-size:13px}}td,th{{padding:8px 10px;text-align:right;border-bottom:1px solid #dae3e8}}td:first-child,th:first-child{{text-align:left}}th{{background:#e8f0f5}}h1,h2{{color:#123f59}}.note{{padding:18px;background:#fff0e6;border-left:4px solid #b46629}}code{{font-size:13px}}</style></head><body>
<h1>Filtered {c['horizon_hours']}h walk-forward study</h1>
<p class="note"><b>Objective not achieved.</b> Primary selected MAE {test.loc['selected','mae']:.6f}, Spearman IC {test.loc['selected','rank_ic']:.6f}. Reconstructed report baseline: MAE {test.loc['baseline','mae']:.6f}, IC {test.loc['baseline','rank_ic']:.6f}. No fixed recipe or registered-weight blend beat both metrics.</p>
<p>Identical full-period evaluation: 18,481 origins from 1 January 2024 through 22 January 2025. Quarterly outer refits; each decision uses two preceding quarterly validation folds. Training labels and selection labels must mature strictly before their respective boundaries. {r['new_fits']} new fits; {r['reused_fits']} exact-population cached fits.</p>
<h2>Overall primary scores</h2>{table(scores.query("model in ['selected','baseline']"))}
<p>Training is in-sample, pooled over {r['training_model_rows']} model-row observations ({r['training_unique_rows']} unique timestamps), using common training support for blend components. Validation is selection-biased: {r['validation_model_rows']} observations ({r['validation_unique_rows']} unique timestamps). Test timestamps occur once.</p>
<h2>What was tested</h2><ul><li>3-, 6-, 12-month and expanding training windows.</li><li>Flow, volatility and funding: Spearman correlation, KS high/low-target distribution separation, feature-quintile magnitude lift, temporal stability and redundancy filtering.</li><li>At most three raw additions per family, six engineered additions and two matured-history additions.</li><li>Absolute activity, causal 1h changes, previous24h means and innovations; training-only clipping, median/IQR scaling and signed logarithms.</li><li>Target normalization by a known matured30d magnitude scale, with scale-proportional sample weights preserving the raw-unit MAE objective.</li><li>Family addition/removal comparisons, conservative blends and a baseline fallback.</li></ul>
<p>KS statistics and lift are descriptive training screens, not IID significance claims or proof of incremental predictive value. All recipe choices use earlier validation; no test-score retuning.</p>
<h2>Did engineering help?</h2>{table(pd.DataFrame(changes))}
<p>These are fixed-recipe test comparisons. Engineering helped its matched precursor but did not beat the reference. Positive improvement means lower MAE or higher IC. Full validation-only contribution comparisons are below.</p>
<h2>Outer decisions</h2>{table(selection)}
<h2>Full test scoreboard</h2>{table(test.reset_index().sort_values('mae').drop(columns=['split']))}
<h2>Monthly primary and baseline scores</h2>{table(monthly.query("model in ['selected','baseline']"))}
<h2>Validation contribution checks</h2>{table(contrib)}
<h2>Training feature-screen counts</h2>{table(pd.DataFrame(admission))}
<p>Passing a screen only grants eligibility. The cap and redundancy check further reduce the model inputs. Per-fit actual columns are in fit_ledger.json.</p>
<h2>Evidence limits and reproduction</h2><p>Report comparator is reconstructed: full MAE181.525993 versus PDF181.523868; original prediction hashes are unavailable. Its IC is reconstructed, not independently verified from original artifacts. All data are historically exposed. Nominal14-day-block intervals do not adjust for the cumulative adaptive research history.</p>
<p>Only48h was executed on real data. Other horizons require canonical price CSVs and separate fits. See USAGE.json. All fitted artifacts and failed results are retained; the original baseline is not replaced.</p>
<p><a href="scores.csv">Scores</a> · <a href="monthly_scores.csv">Monthly scores</a> · <a href="validation_contributions.csv">Validation contributions</a> · <a href="fixed_blend_diagnostics.csv">All fixed-blend diagnostics</a> · <a href="VERIFICATION.json">Audit</a> · <a href="USAGE.json">Usage</a></p>
</body></html>"""
(out/'RESULTS.html').write_text(page,encoding='utf-8');print('Report generated.')
