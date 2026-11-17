"""Render saved residual-correction results; no training."""
from pathlib import Path
import sys,json
import pandas as pd
out=Path(sys.argv[1]);r=json.loads((out/'RESULT.json').read_text());c=json.loads((out/'CONTRACT.json').read_text())
scores=pd.read_csv(out/'scores.csv');monthly=pd.read_csv(out/'monthly_scores.csv');admission=pd.read_csv(out/'feature_admission.csv')
choices=[json.loads(p.read_text()) for p in sorted((out/'outer').glob('*_selection.json'))]
selection=pd.DataFrame([dict(outer=s['outer'],features='; '.join(s['selected']['features']) or 'None: retain baseline',weight=s['selected']['weight']) for s in choices])
def table(f):return f.to_html(index=False,border=0,float_format=lambda v:f'{v:.6f}')
source_text=('Absolute price-volatility features were tested using verified canonical five-minute returns from connected Drive. All 48,096 original 48h labels were reproduced with maximum error9.1e-13logbps. Return timestamps were moved to their endpoints; features exclude endpoints at or after each forecast time. The added inputs did not change the selected forecasts.' if c.get('returns_5m') else 'No canonical price series was supplied in this run. Absolute price-volatility generation was implemented and synthetic-prefix-tested, but was not included. Matured48h outcome summaries are not price-based realized volatility.')
source_link='<a href="CANONICAL_SOURCE_VERIFICATION.json">Canonical source verification</a>' if c.get('returns_5m') else ''

page=f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Incremental residual correction results</title>
<style>body{{font:16px/1.5 Segoe UI,sans-serif;max-width:1150px;margin:40px auto;padding:0 24px;background:#fafcfd;color:#1c3040}}table{{width:100%;border-collapse:collapse;background:white;font-size:13px;margin:20px 0 30px}}th,td{{padding:9px;border-bottom:1px solid #dce5eb;text-align:right;overflow-wrap:anywhere}}th{{background:#e9f0f5}}td:first-child,th:first-child{{text-align:left}}.note{{padding:18px;background:#fff1df;border-left:4px solid #b87925}}</style></head><body>
<h1>{c['horizon_hours']}h baseline plus incremental residual correction</h1>
<p class="note"><b>The joint objective was not achieved.</b> Test MAE 181.865736 versus reconstructed reference 181.525993; Spearman IC 0.084654 versus 0.077390. Higher observed IC came with worse MAE. Nominal paired-block intervals include zero for both differences; do not present this as a proven improvement.</p>
<p>Full 18,481 baseline-period timestamps, January 2024 through 22 January 2025. This result belongs to the complete walk-forward correction policy, not standalone CatBoost or LightGBM.</p>
<h2>What changed</h2><ul><li>Correction targets use genuinely chronological out-of-fold baseline errors, not in-sample baseline residuals.</li><li>Feature screening uses only data before the earliest inner validation fold. Each shortlisted feature then needs individual incremental MAE and IC improvement across earlier walk-forward predictions.</li><li>Three two-month inner folds; at least4/6 monthly MAE wins and positive mean monthly IC gain required.</li><li>At most 3 admitted features. Three-bin conditional median errors are shrunk toward the global median error using 2048 pseudo-observations.</li><li>Correction limited to 50 bps before weight; weights 0,.1,.25,.5; final change at most 25 bps. Zero correction remains available.</li></ul>
<h2>Overall scores</h2>{table(scores)}
<p>Training here means correction in-sample on historical OOF baseline errors, not baseline in-sample. Training has {r['training_model_rows']} model-row observations ({r['training_unique_rows']} unique timestamps); selection validation has {r['validation_model_rows']} observations ({r['validation_unique_rows']} unique). Test timestamps occur once. Pooled validation IC can differ from within-decision gains because windows overlap and selected corrections differ.</p>
<h2>Selected correction by period</h2>{table(selection)}
<h2>Monthly results</h2>{table(monthly)}
<h2>Individual feature-admission results</h2>{table(admission)}
<h2>Data and evidence limits</h2><p>Only 48h was executed on real data. {source_text} {source_link} Canonical source definition is cross-venue trade VWAP, total quote volume divided by total base volume.</p>
<p>Fresh confirmation was not run: existing historical development data have already been inspected. The comparator is reconstructed (181.525993), not exact original report artifact replay (181.523868). All earlier experiments remain unchanged.</p>
<p><a href="VERIFICATION.json">Independent audit</a> · <a href="scores.csv">Overall scores</a> · <a href="feature_admission.csv">Feature admission</a> · <a href="monthly_scores.csv">Monthly scores</a> · <a href="USAGE.json">Usage and missing-data requirements</a></p>
</body></html>"""
(out/'RESULTS.html').write_text(page,encoding='utf-8');print('Report generated.')
