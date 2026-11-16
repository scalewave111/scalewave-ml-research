"""Create the 16h report from three completed runs. No training or inference."""
from pathlib import Path
import json
import hashlib
import shutil
import numpy as np
import pandas as pd
import joblib
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

ROOT = Path(__file__).resolve().parents[1]
PIPE = ROOT/'Code/experiments/excursion_pipeline_16h_v1/20261113T104606735810Z'
SEARCH = PIPE/'search/20261113T104614228710Z'
FOLLOW = PIPE/'price_followup/20261113T110511390796Z'
TUNE = ROOT/'Code/experiments/excursion_tuning_16h_v1/20261113T113009880786Z'
OUT = ROOT/'Report/output/pdf/ScaleWave_16h_Excursion_Models_Report.pdf'
SUP = ROOT/'Report/sixteen_hour_excursion_report_sources'
SUP.mkdir(parents=True, exist_ok=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
TARGETS = ['up_magnitude_log_bps', 'down_magnitude_log_bps']
ISSUES = ['2024-01', '2024-04', '2024-07', '2024-10']

def js(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def side(target): return 'UP' if target.startswith('up_') else 'DOWN'
def metric(y, p):
    return [np.abs(y-p).mean(), np.sqrt(np.square(y-p).mean()), pd.Series(y).corr(pd.Series(p), method='spearman')]

manifest, verified = {}, []
frames = {}
for name, folder in [('search', SEARCH), ('follow', FOLLOW), ('tune', TUNE)]:
    done = js(folder/'COMPLETE.json')
    assert done['status'] == 'COMPLETE' and done['replay_passed']
    assert sha(folder/'predictions.csv') == done['predictions_sha256']
    f = pd.read_csv(folder/'predictions.csv')
    assert len(f) == 36962 and not f.duplicated(['decision_time', 'target']).any()
    frames[name] = f
    for fn in ['CONTRACT.json', 'COMPLETE.json', 'predictions.csv', 'selections.json']:
        manifest[str(folder/fn)] = sha(folder/fn)

a, b, c = [frames[n].sort_values(['target', 'decision_time']).reset_index(drop=True) for n in ['search','follow','tune']]
for other in [b, c]:
    assert a[['target','decision_time','issue']].equals(other[['target','decision_time','issue']])
    np.testing.assert_allclose(a.actual, other.actual, rtol=0, atol=1e-10)
np.testing.assert_allclose(a.prediction, b.original, rtol=0, atol=1e-7)
np.testing.assert_allclose(b.selected, c.previous, rtol=0, atol=1e-7)
ms = pd.read_csv(SEARCH/'metrics.csv')
mf = pd.read_csv(FOLLOW/'metrics_all_variants.csv')
mt = pd.read_csv(TUNE/'metrics.csv')

# Independently reproduce every saved metric, including partial-coverage rows.
for name, metrics in [('search', ms), ('follow', mf), ('tune', mt)]:
    f = frames[name]
    for _, row in metrics.iterrows():
        g = f[f.target.eq(row.target)]
        if row.period != 'ALL': g = g[g.period.eq(row.period)]
        g = g[g[row.model].notna()]
        assert len(g) == row.rows
        vals = metric(g.actual.to_numpy(), g[row.model].to_numpy())
        np.testing.assert_allclose(vals, [row.mae, row.rmse, row.ic], rtol=1e-9, atol=1e-8, equal_nan=True)
        verified.append(dict(stage=name, target=row.target, period=row.period, model=row.model, rows=len(g)))

ss, sf, st = [js(folder/'selections.json') for folder in [SEARCH, FOLLOW, TUNE]]
lookup = lambda selections, issue, target: next(s for s in selections if s['issue'].startswith(issue) and s['target']==target)
original = sorted({f for s in ss for f in s['features']})
ids = {f:f'F{i+1:02d}' for i,f in enumerate(original)}
groups = js(FOLLOW/'CONTRACT.json')['groups']
groups['price_16h'] = [f'NEWPRICE__{m}__16h' for m in ['rv','up_rv','down_rv','net','past_range','distance_from_high','distance_from_low','max_abs_return']]
features_export, params_export = [], []
for issue in ISSUES:
    for target in TARGETS:
        key = f'{issue}_{target}'
        s1, s2, s3 = [lookup(sel, issue, target) for sel in [ss, sf, st]]
        assert s1['model'] == 'tree'
        plan = js(FOLLOW/key/'selection.json')
        spec = plan['variants'][plan['selected_variant']]
        assert spec['features'] == s2['features'] and spec['model']=='tree'
        path = FOLLOW/key/spec['artifact']
        assert sha(path) == spec['artifact_sha256']
        bundle = joblib.load(path)
        params_export.append(dict(issue=issue, target=target, branch='previous',
            raw_inputs=len(s2['features']), n_features_in=int(bundle['model']['model'].n_features_in_),
            iterations=int(bundle['model']['model'].n_iter_), params=bundle['model']['model'].get_params()))
        path = TUNE/key/'model.joblib'
        assert sha(path) == js(TUNE/key/'MODEL_HASH.json')['sha256']
        tuned = joblib.load(path)
        assert tuned['control_features'] == s2['features'] and tuned['selection'] == s3['selected']
        if tuned['candidate'] is not None:
            params_export.append(dict(issue=issue, target=target, branch='candidate',
                raw_inputs=len(s3['selected']['features']), n_features_in=int(tuned['candidate'].n_features_in_),
                iterations=int(tuned['candidate'].n_iter_), params=tuned['candidate'].get_params()))
        for stage, cols in [('initial',s1['features']),('previous',s2['features']),
                            ('candidate',s3['selected']['features'] if s3['selected'] else [])]:
            features_export += [dict(issue=issue,target=target,stage=stage,feature=f) for f in cols]
        for folder in [SEARCH/key,FOLLOW/key,TUNE/key]:
            for fn in ['selection.json','validation_trials.csv']:
                if (folder/fn).exists(): manifest[str(folder/fn)] = sha(folder/fn)
        manifest[str(path)] = sha(path)

pd.DataFrame(features_export).to_csv(SUP/'exact_feature_membership.csv',index=False)
(SUP/'fitted_model_parameters.json').write_text(json.dumps(params_export,indent=2,default=str))
pd.DataFrame(verified).to_csv(SUP/'metric_reproduction_audit.csv',index=False)
for folder in [SEARCH,FOLLOW,TUNE]:
    for path in folder.glob('*.py'): manifest[str(path)] = sha(path)
    for name in ['metrics.csv','metrics_all_variants.csv']:
        if (folder/name).exists(): manifest[str(folder/name)] = sha(folder/name)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Body16',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=9,textColor=colors.HexColor('#25364a')))
styles.add(ParagraphStyle(name='Cell16',fontName='Helvetica',fontSize=8,leading=10.4,splitLongWords=True))
styles.add(ParagraphStyle(name='Small16',fontName='Helvetica',fontSize=8,leading=11,spaceAfter=7,textColor=colors.HexColor('#426077')))
styles.add(ParagraphStyle(name='Heading16',fontName='Helvetica-Bold',fontSize=22,leading=26,spaceAfter=12,textColor=colors.HexColor('#16334b')))
story=[]
def p(text,style='Body16'): return Paragraph(text,styles[style])
def para(text): story.append(p(text))
def title(text):
    if story: story.append(PageBreak())
    story.extend([p('SCALEWAVE  /  BTCUSDT  /  16-HOUR EXCURSIONS','Small16'),p(text,'Heading16'),Spacer(1,5)])
def table(headers, rows, widths):
    data=[[p(escape(str(c)).replace('\n','<br/>'),'Cell16') for c in row] for row in [headers]+rows]
    tab=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce9f4')),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#607b95')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f3f6f9')])]))
    story.extend([tab,Spacer(1,9)])
def num(v): return f'{v:.4f}'
def get(metrics,target,model,period='ALL'):
    return metrics[(metrics.target==target)&(metrics.model==model)&(metrics.period==period)].iloc[0]
def fids(cols): return ', '.join(ids[c] for c in cols if c in ids) or 'None'
def blocks(cols):
    new={c for c in cols if c.startswith('NEWPRICE')}
    found=[]
    for hours in [1,4,12,16,24,48,168]:
        g=set(groups[f'price_{hours}h'])
        if g <= new: found.append(f'{hours}h'); new-=g
    ratios=set(groups['volatility_ratios'])
    has_ratios=ratios<=new
    if has_ratios: new-=ratios
    assert not new
    return ('P['+', '.join(found)+']' if found else '')+(' + R' if has_ratios and found else ('R' if has_ratios else '')) or 'None'
def label(model):
    return {'training_mean':'Training mean baseline','training_median':'Training median baseline',
      'original':'Stage 1: selected F4Z model','selected':'Stage 2: selected price strategy',
      'tuned':'Stage 3: selected tuned blends','all_price_plus_original':'Original + all 51 price inputs',
      'price_only':'All 51 price inputs only','remove_original_features':'Remove F4Z from greedy endpoint',
      'remove_volatility_ratios':'Remove ratios from greedy endpoint'}.get(model,
        model.replace('add_price_','Original + price ').replace('add_volatility_ratios','Original + ratios'))

title('Models, features and ranked results')
para('Prepared 13 November 2026. This report covers <b>three completed 16-hour experiments</b>: feature search, price-feature follow-up and model tuning. Separate UP and DOWN models predict the largest upward and downward reference-path excursions over the next 960 minutes.')
table(['Target','Completed strategy','MAE','RMSE','Rank IC'],[[side(t),name,num(r.mae),num(r.rmse),num(r.ic)]
    for t in TARGETS for name,r in [('Stage 1: F4Z search',get(ms,t,'prediction')),('Stage 2: price follow-up',get(mf,t,'selected')),('Stage 3: tuned blends',get(mt,t,'selected'))]],[45,190,85,85,95])
para('<b>Latest selected strategy:</b> UP MAE <b>104.9636</b>, DOWN MAE <b>108.1573</b> log bps. Relative to Stage 2, MAE fell by 0.66% and 0.47%; RMSE also fell, to 157.0264 and 171.6461. Six of eight quarterly side models use a 50/50 blend; two retain Stage 2 unchanged.')
para('<b>Ranking distinction:</b> the lowest observed MAE across full-coverage diagnostic variants is UP original + all 51 price inputs (104.8815), and DOWN removal of F4Z from the greedy endpoint (107.1630). Those are retrospective comparisons, not replacements selected by the earlier-validation policy. The tuned strategy has lower RMSE than either respective MAE leader.')
table(['Coverage','Verified scope'],[['Origins per target','18,481 evaluation timestamps; 36,962 target rows per completed run'],['Dates','January 2024 through 22 January 2025; original half-hour sample'],['Training / fitting','29,615 original initial keys; Jan/Apr/Jul/Oct 2024 fits'],['Report checks','Three prediction hashes, matching keys/truths, carried controls and all saved metric rows checked; fitted model settings inspected'],['Units','MAE and RMSE: log bps. Rank IC: dimensionless.']],[145,355])
para('The historical evaluation has been used repeatedly. This report describes measured results, not independent confirmation. No new training was performed for the report.')

title('Targets and chronological evaluation')
para('<b>UP = 10,000 x ln(maximum future path price / origin price).</b><br/><b>DOWN = 10,000 x ln(origin price / minimum future path price).</b>')
para('The canonical one-minute reference-price path includes the origin and ends 16 hours later. Both targets are nonnegative; all valid rows, including zeros, enter both models. They measure path extremes, not the signed return at the endpoint. Inputs and targets are BTCUSDT.')
table(['Rule','Implementation'],[['Initial data','Original 29,615 timestamps: 22 Apr 2022 00:30 to 29 Dec 2023 23:30 UTC. Original 48-hour train/development boundary gap remains absent.'],['Outer fits','1 Jan, 1 Apr, 1 Jul, 1 Oct 2024. Each predicts the following quarter. October also predicts January 2025, reported separately.'],['Strict maturity','An origin enters training only if origin + 16 hours < fit boundary. Equality is excluded.'],['Earlier validation','Three 3-month windows starting 9, 6 and 3 months before each outer fit. Validation labels must also finish strictly before their own window end.'],['Recent-year candidates','Eligible training origins additionally satisfy origin >= boundary - 365 days. The same 16-hour maturity rule still applies.'],['No dropped evaluation inputs','Missing features do not remove evaluation rows. Stage 1/2 impute using training data; Stage 3 candidates use native missing-value routing.']],[135,365])
table(['Fit','Validation periods','Prediction period'],[['Jan 2024','Apr-Jun, Jul-Sep, Oct-Dec 2023','Jan-Mar 2024'],['Apr 2024','Jul-Sep, Oct-Dec 2023; Jan-Mar 2024','Apr-Jun 2024'],['Jul 2024','Oct-Dec 2023; Jan-Mar, Apr-Jun 2024','Jul-Sep 2024'],['Oct 2024','Jan-Mar, Apr-Jun, Jul-Sep 2024','Oct-Dec 2024 and Jan 2025']],[70,250,180])
para('All dates are UTC. Windows are described by calendar months; their final 16-hour labels are purged. Earlier evaluation rows can enter later walk-forward training once their labels mature.')

title('Model architecture')
para('The selected base family is <b>HistGradientBoostingRegressor with absolute-error loss</b>. Small trees are added sequentially to reduce errors; predictions are median-oriented. UP and DOWN are fitted independently. No direction classifier or probability weighting is involved.')
def architecture():
    d=Drawing(500,235)
    def box(x,y,w,h,lines,fill='#eaf1f7'):
        d.add(Rect(x,y,w,h,rx=5,ry=5,fillColor=colors.HexColor(fill),strokeColor=colors.HexColor('#8095a6')))
        for k,line in enumerate(lines): d.add(String(x+w/2,y+h-18-k*14,line,textAnchor='middle',fontName='Helvetica',fontSize=9,fillColor=colors.HexColor('#16334b')))
    def arrow(x1,y1,x2,y2):
        d.add(Line(x1,y1,x2,y2,strokeColor=colors.HexColor('#607b95')))
        d.add(Polygon([x2,y2,x2-3,y2+6,x2+3,y2+6],fillColor=colors.HexColor('#607b95')))
    box(145,190,210,38,['Past features at forecast origin'])
    d.add(Line(250,190,115,170));d.add(Line(250,190,385,170))
    box(0,108,230,62,['Stage 2 control branch','Training medians + missing flags','Training-only scaling -> 80 trees'])
    box(270,108,230,62,['New Stage 3 candidate branch','Raw inputs; native NaN handling','120 / 160 / 200 boosted trees'])
    arrow(115,108,115,83);arrow(385,108,385,83)
    box(0,44,230,39,['Clip control prediction at zero'])
    box(270,44,230,39,['Clip candidate prediction at zero'])
    d.add(Line(115,44,250,25));d.add(Line(385,44,250,25))
    box(105,0,290,27,['Final = 0.5 x control + 0.5 x candidate'],'#dcebdc')
    return d
story.extend([architecture(),Spacer(1,12)])
para('The diagram shows all six accepted tuned changes. The other two fits retain the control with weight 1. The saved run tested pure-candidate weight 1 and blend weight 0.5; every accepted change used 0.5.')
table(['Component','Role in the experiments'],[['Histogram boosting','Selected in all eight Stage 1 fits and all eight Stage 2 fits; also every accepted Stage 3 candidate.'],['Ridge regression','Stage 1 candidate, fitted with squared error and selected using MAE. Never selected for these outer fits; no standalone outer Ridge predictions saved.'],['Depth-2 decision tree','Single-feature screening only; not an outer forecasting model.'],['Mean / median constants','Training-label baselines, updated each fit. Their pooled IC can be nonzero because constants change across quarters.']],[145,355])
para('Stage 1/2 append one missing indicator per raw feature, doubling their model input dimension. Stage 3 candidates retain the raw feature count. A blend has two independent branches, not one concatenated feature vector.')

title('Hyperparameters and preprocessing')
table(['Setting','Control','Small','Medium','Flexible'],[['Loss','absolute_error','absolute_error','absolute_error','absolute_error'],['Iterations','80','120','160','200'],['Learning rate','0.05','0.05','0.05','0.05'],['Maximum depth','3','2','3','4'],['Maximum leaves','7','4','7','15'],['Minimum leaf samples','128','128','96','64'],['L2 regularization','1','10','5','10'],['Early stopping','False','False','False','False'],['Random seed','48','48','48','48'],['Maximum bins','255','255','255','255'],['Missing-value handling','Median + flags','Native NaN','Native NaN','Native NaN']],[155,86,86,86,87])
para('Selected settings are verified against saved fitted estimators; the unselected small configuration is documented in the saved contract and code. All training scripts use a two-thread limit. The three candidate configurations are not separately saved full-evaluation strategies. Selected settings appear in the cycle tables.')
table(['Other setting','Exact behavior'],[['Control preprocessing','Training median per column (all-missing median replaced by 0); append missing flags; center and scale using training mean/standard deviation. Standard deviation < 1e-8 is replaced by 1.'],['Candidate preprocessing','Histogram bin boundaries learned on training inputs; native missing values. No imputation, missing flags or explicit scaling.'],['Output constraint','Clip each model prediction to max(0, prediction), then blend if selected.'],['Ridge search candidate','alpha = 0.01 x eligible training row count; same training-only preprocessing as control.'],['Screening tree','DecisionTreeRegressor: absolute_error, max_depth=2, min_samples_leaf=128, random_state=48. Up to 4,096 evenly spaced training rows; all validation rows scored.']],[150,350])
para('The L1 objective favors typical absolute error. It does not directly minimize RMSE; Stage 3 therefore adds a validation MSE guardrail. A passing guardrail cannot guarantee lower errors on later observations.')

title('How features and settings were selected')
table(['Stage','Search and acceptance rules'],[['1 - F4Z search','522 masked F4Z candidates. In every inner training fold: >=80% finite coverage, >=1,000 finite values and variation. Rank single-feature MAE trees; remove redundancy at absolute Spearman >=0.90 using earliest training only. Shortlist up to 16. Separate Ridge and boosting forward/removal searches, up to 6 features; no mandatory old columns.'],['1 - Decision','Each addition needs >=0.1% pooled MAE improvement and wins in >=2 of 3 validation windows. Removal accepted if no worse pooled and in >=2 windows. Best eligible family must beat median; otherwise median fallback.'],['2 - Price follow-up','Keep selected original features as control. Try up to two greedy price-block additions, all 51 price inputs + original, price-only, and group removals. Alternative must beat original and median by >=0.1% pooled MAE and >=2/3 wins against each. Otherwise retain original.'],['3 - Tuning packs','(a) Stage 2 selected inputs; (b) those inputs + eight 16h measures; (c) F4Z retained in Stage 2 + all 59 price inputs. No mandatory recovery of F4Z previously removed.'],['3 - Grid','3 packs x 3 configurations x 2 histories = 18 candidates per side/fit; each has three validation fits. Score candidate weights 0.5 and 1.0 against control: 36 alternatives.'],['3 - Decision','Require >=0.1% pooled MAE reduction against both control and median, >=2/3 MAE wins against each, and pooled MSE no worse than control. Choose lowest eligible validation MAE; otherwise keep control.']],[120,380])
para('Pooled validation errors weight each window by its row count. Final selection is frozen before scoring the associated outer quarter. The recent-year option trims origins, not labels or feature lookbacks.')
para('The reference feature lists were already selected using the same earlier validation windows. Further search adaptively reuses them; this is not fully nested validation. The later history has also been examined repeatedly. Candidate tests and period rankings are exploratory evidence.')

title('How the new price features were created')
para('Source: checksum-verified canonical nonoverlapping five-minute BTC returns. Each r is <b>10,000 x ln(P_end/P_start)</b>, in log bps, indexed at its endpoint. For a forecast at t, feature windows use endpoints <b>t - h hours &lt;= endpoint &lt; t</b>. The return ending exactly at t is excluded.')
table(['Measure / exact suffix','Formula over the past window'],[['rv','sqrt(sum(r squared))'],['up_rv','sqrt(sum(max(r, 0) squared))'],['down_rv','sqrt(sum(min(r, 0) squared))'],['net','sum(r)'],['past_range','max(L) - min(L)'],['distance_from_high','max(L) - latest L'],['distance_from_low','latest L - min(L)'],['max_abs_return','max(abs(r))']],[175,325])
para('L is the cumulative log-bps close-path level across the window, including its starting level. Differences are invariant to the arbitrary starting level. Ranges and distances describe five-minute closes, not intrabar exchange high/low prices.')
para('<b>Stage 2:</b> eight measures at 1h, 4h, 12h, 24h, 48h and 168h = 48 features, plus three ratios = <b>51</b>.<br/><b>Stage 3:</b> the same eight measures at 16h add eight more = <b>59</b>. The 16h forecast horizon does not force every input to use a 16h lookback.')
para('<b>Volatility ratios:</b> for pairs (1,24), (4,24), (24,168) hours:<br/>(RV_short / sqrt(short hours)) / max(RV_long / sqrt(long hours), 0.000001).<br/>These are dimensionless; the eight raw measures are in log bps.')
para('A complete h-hour window requires exactly 12 x h valid five-minute returns: 192 for 16h. Incomplete windows remain missing. The causal audit checks prefix equality and changes future returns to verify that future data cannot change a current feature.')

title('Exact new-feature names and example')
table(['Block','Exact column naming'],[['P[h]: eight inputs','NEWPRICE__rv__{h}h\nNEWPRICE__up_rv__{h}h\nNEWPRICE__down_rv__{h}h\nNEWPRICE__net__{h}h\nNEWPRICE__past_range__{h}h\nNEWPRICE__distance_from_high__{h}h\nNEWPRICE__distance_from_low__{h}h\nNEWPRICE__max_abs_return__{h}h'],['Allowed h values','1, 4, 12, 16, 24, 48, 168. The 16h block appears only in Stage 3 candidates.'],['R: three ratios','NEWPRICE__rv_ratio__1h_24h\nNEWPRICE__rv_ratio__4h_24h\nNEWPRICE__rv_ratio__24h_168h']],[120,380])
para('In the membership tables, <b>P[4h, 16h]</b> means all eight measures for each listed lookback (16 inputs). <b>R</b> always means all three ratio columns. This notation exactly defines every new column; there is no unnamed subset.')
para('<b>Worked arithmetic example:</b> take four five-minute returns [10, -5, 20, -15] log bps. Their relative cumulative path is [0, 10, 5, 25, 10]. This short illustration is for arithmetic only; actual one-hour and 16-hour windows require 12 and 192 returns.')
table(['Feature','Calculation','Value (log bps)'],[['RV','sqrt(100 + 25 + 400 + 225)','27.3861'],['Up RV','sqrt(100 + 400)','22.3607'],['Down RV','sqrt(25 + 225)','15.8114'],['Net','10 - 5 + 20 - 15','10'],['Past range','25 - 0','25'],['Distance from high','25 - 10','15'],['Distance from low','10 - 0','10'],['Maximum absolute return','max(10, 5, 20, 15)','20']],[150,245,105])
para('Example ratio: if 1h RV = 20 and 24h RV = 120, normalized ratio = (20 / sqrt(1)) / (120 / sqrt(24)) = <b>0.8165</b>. A raw 20/120 ratio would omit the duration adjustment.')

for start in range(0,len(original),10):
    title('Original F4Z feature dictionary'+(' - continued' if start else ''))
    para('These IDs identify exact original feature names across all three stages. They are identifiers for compact tables, not feature ranks. Source suffixes distinguish percentile, robust score (RZ) and historical active rate; these are not interchangeable.')
    table(['ID','Exact feature name'],[[ids[f],f] for f in original[start:start+10]],[38,462])
    para('Feature columns are inherited from the masked package. Historical activity rates describe past activity; signed flow acceleration is distinct from total flow activity. Original source availability and provisional/funding timing limitations remain inherited.')

title('Stage 1 and Stage 2 feature membership')
para('All rows below use the 80-iteration control architecture. Raw feature counts exclude missing indicators. P and R are defined in the new-feature section; F IDs resolve to the exact dictionary names.')
table(['Fit','Side','Stage 1 original IDs','Count'],[[i,side(t),fids(lookup(ss,i,t)['features']),len(lookup(ss,i,t)['features'])] for i in ISSUES for t in TARGETS],[65,40,355,40])
table(['Fit','Side','Stage 2 F4Z IDs','Price blocks','Raw count'],[[i,side(t),fids(s['features']),blocks(s['features']),len(s['features'])] for i in ISSUES for t in TARGETS for s in [lookup(sf,i,t)]],[60,40,180,175,45])
para('Stage 2 October DOWN removed all original F4Z features and uses P[1h] + R. October UP retains three F4Z features and P[4h] + R. These control branches remain present in the final October blends.')

for target in TARGETS:
    title(f'Stage 3: {side(target)} models and exact inputs')
    para('Each row is the final policy chosen using earlier validation. Control always means the corresponding Stage 2 model above. Candidate counts describe the new branch only; the blend preserves both branches.')
    rows=[]
    for issue in ISSUES:
        old=lookup(sf,issue,target)
        selection=lookup(st,issue,target)['selected']
        if selection is None:
            rows.append([issue,'Retain control','All mature history',fids(old['features']),blocks(old['features']),f'{len(old["features"])} control'])
        else:
            rows.append([issue,selection['config']+'; 50/50 blend', 'Last 365 days' if selection['days'] else 'All mature history',fids(selection['features']),blocks(selection['features']),str(len(selection['features']))])
    table(['Fit','Selected model','Candidate history','Candidate F4Z IDs','Candidate price blocks','Raw inputs'],rows,[60,100,85,90,120,45])
    latest=lookup(st,'2024-10',target)['selected']
    old=lookup(sf,'2024-10',target)
    para('<b>Latest October model, also used for January 2025:</b>')
    table(['Branch','Exact inputs','Raw / fitted dimension','Trees'],[['Control (50%)',fids(old['features'])+'; '+blocks(old['features']),f'{len(old["features"])} / {2*len(old["features"])}','80'],['Candidate (50%)',fids(latest['features'])+'; '+blocks(latest['features']),f'{len(latest["features"])} / {len(latest["features"])}','200']],[95,235,100,70])
    para('Final prediction = 0.5 x nonnegative control prediction + 0.5 x nonnegative candidate prediction. Both October branches use all mature history; the candidate has flexible settings: depth 4, up to 15 leaves, minimum leaf 64, L2 = 10, learning rate 0.05.')
    if target.startswith('up'):
        para('UP keeps the Stage 2 control in January. April uses a recent-year flexible candidate with 27 raw inputs; July and October use all-history flexible candidates with 62. The October blend has 14 raw control inputs (28 after preprocessing) and 62 native candidate inputs; it is not one 76-column model.')
    else:
        para('DOWN retains the Stage 2 control in July. January uses a medium all-history candidate with 64 raw inputs; April uses a medium recent-year candidate with 65. October uses only price features: the 11-input control and the 19-input candidate (P[1h, 16h] + R).')

ranked=[]
for target in TARGETS:
    title(f'{side(target)}: full-coverage model ranking')
    rows=mf[(mf.target==target)&(mf.period=='ALL')&mf.full_period_coverage.eq(True)&mf.rows.eq(18481)].copy()
    added=get(mt,target,'selected').to_dict();added['model']='tuned'
    rows=pd.concat([rows,pd.DataFrame([added])],ignore_index=True)
    rows['MAE_rank']=rows.mae.rank(method='min').astype(int)
    rows['RMSE_rank']=rows.rmse.rank(method='min').astype(int)
    rows['IC_rank']=rows.ic.rank(ascending=False,method='min').astype(int)
    rows=rows.sort_values('mae')
    ranked.append(rows)
    para(f'All {len(rows)} rows use the identical 18,481 timestamps and canonical 16-hour target. Primary rank: lower MAE. Parentheses show separate RMSE and IC ranks; higher IC is better. These are model/feature policies spanning four quarterly fits.')
    table(['MAE rank','Model / strategy','MAE','RMSE (rank)','IC (rank)'],[[r.MAE_rank,label(r.model),num(r.mae),f'{r.rmse:.4f} ({r.RMSE_rank})',f'{r.ic:.4f} ({r.IC_rank})'] for r in rows.itertuples()],[42,213,75,93,77])
    para('Stage 1, Stage 2 and Stage 3 are earlier-validation-selected strategies. Other feature variants are diagnostic outer scores; ranking them after seeing these results is hindsight selection. Mean/median are baseline controls. Partial-coverage variants are not assigned a pooled rank.')
    para('Removal variants remove a group from each quarter\'s greedy-search endpoint, so their retained inputs can vary by quarter. The price-only row always uses all 51 Stage 2 price inputs. Tuning candidates that were not selected have only validation scores, so they cannot be ranked as full-evaluation models.')

title('Evaluation metrics and their meaning')
table(['Metric','Definition','How to read it'],[['MAE','mean(abs(actual - prediction))','Lower is better. Main selection and ranking objective; average absolute error in log bps.'],['RMSE','sqrt(mean((actual - prediction) squared))','Lower is better. Penalizes large errors more heavily; also in log bps.'],['MSE','mean((actual - prediction) squared)','Squared log-bps units. Stage 3 acceptance guardrail; RMSE has the same ordering on identical rows.'],['Rank IC','Spearman correlation between actual and predicted values','Higher is better; between -1 and 1. Measures ordering, not calibration or absolute accuracy.'],['MAE gain','Previous MAE - new MAE','Positive means improvement. Percentage reduction = gain / previous MAE x 100.']],[82,200,218])
para('Example: true excursion 150 log bps and forecast 110 gives absolute error 40, squared error 1,600. MAE averages absolute errors across origins; RMSE takes the square root only after averaging squared errors. Pooled RMSE is not the arithmetic average of quarterly RMSEs.')
para('Log bps are not ordinary percentage points. For an upward excursion u, simple percentage = 100 x (exp(u/10,000) - 1). For a downward excursion magnitude d, the positive drop percentage = 100 x (1 - exp(-d/10,000)). Errors in log bps should not be relabeled as percentage errors.')
table(['Comparison','UP','DOWN'],[['Stage 3 MAE','104.9636','108.1573'],['Stage 2 to Stage 3 MAE reduction','0.7000 (0.66%)','0.5123 (0.47%)'],['Stage 3 RMSE','157.0264','171.6461'],['Training mean baseline RMSE','154.8614','170.1982']],[270,115,115])
para('The latest selected models beat the median baseline on MAE, but still lose to the mean baseline on RMSE. The improvement therefore concerns typical absolute error; large-error performance remains a separate limitation. Reported ranks do not establish statistical significance.')

title('Results by evaluation period')
for target in TARGETS:
    para('<b>'+side(target)+' - Stage 3 selected policy</b>')
    rows=[]
    for period in ['2024Q1','2024Q2','2024Q3','2024Q4','2025-01','ALL']:
        new,old=get(mt,target,'selected',period),get(mt,target,'previous',period)
        rows.append([period,int(new.rows),num(new.mae),num(new.rmse),num(new.ic),f'{old.mae-new.mae:+.4f}',f'{old.rmse-new.rmse:+.4f}'])
    table(['Period','Rows','MAE','RMSE','IC','MAE gain','RMSE gain'],rows,[65,45,80,80,65,82,83])
para('Gains compare Stage 3 with Stage 2 on the same period; positive values mean lower error. UP retains Q1, slightly worsens Q2 MAE, improves Q3/Q4, and worsens January MAE/RMSE. DOWN improves Q1/Q2, retains Q3, worsens Q4 MAE while improving Q4 RMSE, and improves January on both errors.')
para('The October model remains unchanged in January. These period differences are measured outcomes of the earlier selection policy; no model was chosen using its own outer-period error.')

title('Evidence, checks and interpretation')
table(['Stage','Saved experiment location under Code/experiments'],[['1 - Original search','excursion_pipeline_16h_v1/20261113T104606735810Z/search/20261113T104614228710Z'],['2 - Price follow-up','excursion_pipeline_16h_v1/20261113T104606735810Z/price_followup/20261113T110511390796Z'],['3 - Tuning','excursion_tuning_16h_v1/20261113T113009880786Z']],[125,375])
para('<b>Evidence read:</b> COMPLETE.json, CONTRACT.json, predictions.csv, metrics tables, selections.json, per-fit model cards, fitted model parameters, and the saved source snapshots. Every prediction-file SHA256 matches its completion record. Model-artifact hashes for selected Stage 2 and Stage 3 models also match.')
para(f'<b>Report verification:</b> all {len(verified)} saved metric rows were independently reproduced from saved predictions, including partial-coverage variants. Exact timestamps, actual targets and carried control predictions agree across stages. This report loaded models only to inspect parameters; it performed no training or new model inference. The original runs record successful model replay.')
para('The companion source folder stores full-precision rankings, exact feature membership, fitted parameters, metric-reproduction audit, original metric tables, and source hashes: <b>Report/sixteen_hour_excursion_report_sources/</b>. Partial-coverage variants are saved separately and excluded from full-period ranks.')
para('Only the canonical 16-hour excursion experiments are compared. Other forecast horizons, absolute endpoint returns and signed-return models have different targets and are excluded. Ridge and screening trees lack standalone outer predictions in these runs, so no invented evaluation rank is supplied.')
para('The sampled population, overlapping 16-hour outcomes, repeated historical evaluation and adaptively reused validation windows limit interpretation. Feature inclusion is not causal importance; correlated inputs can substitute for each other. Inherited availability masking and source-construction limitations are not freshly certified by this report.')
para('No guarantee of future improvement, independent significance, trading profitability or transaction-cost performance follows from these historical error rankings.')

pd.concat(ranked).to_csv(SUP/'full_coverage_model_rankings.csv',index=False)
mf[(mf.period=='ALL')&~mf.full_period_coverage.eq(True)].to_csv(SUP/'partial_coverage_variants_unranked.csv',index=False)
for tag, frame in [('search',ms),('followup',mf),('tuning',mt)]: frame.to_csv(SUP/f'{tag}_metrics.csv',index=False)
(SUP/'source_hashes.json').write_text(json.dumps(manifest,indent=2))
def footer(canvas, doc):
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#607b95'))
    canvas.drawString(42,24,'ScaleWave | 16-hour UP/DOWN excursions | 13 Nov 2026')
    canvas.drawRightString(553,24,str(doc.page))
SimpleDocTemplate(str(OUT),pagesize=(595,842),leftMargin=42,rightMargin=42,topMargin=35,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
print(json.dumps(dict(pdf=str(OUT),original_features=len(original),verified_metric_rows=len(verified),ranking_rows=sum(map(len,ranked))),indent=2))
