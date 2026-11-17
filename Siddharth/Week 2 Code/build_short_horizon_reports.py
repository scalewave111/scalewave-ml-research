"""Build two factual reports from saved results only; no model imports or fitting."""
from pathlib import Path
import json, hashlib, sys
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
W1 = ROOT.parent / 'Week 1'
E = W1 / 'Code/experiments'
OUT = ROOT / 'Report/output/pdf'
SUP = ROOT / 'Report/short_horizon_reports_2026-11-11'
OUT.mkdir(parents=True, exist_ok=True)
SUP.mkdir(parents=True, exist_ok=True)
NEW = sorted((ROOT / 'Code/experiments/short_horizon_median_v1').glob('*/COMPLETE.json'))[-1].parent
sources = {}

def track(p):
    p = Path(p)
    sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p

def csv(p): return pd.read_csv(track(p))
def js(p): return json.loads(track(p).read_text(encoding='utf-8'))

features = [
('A','Liquidation historical activity, 5m','F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__5m__value'),
('B','Liquidation historical activity, 15m','F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__15m__value'),
('C','Liquidation historical activity, 1h','F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value'),
('D','Price volatility ratio, 4h / 24h','PRICE__vol_ratio_4h_24h'),
('E','Price realized volatility, 168h (7 days)','PRICE__realized_vol_168h'),
('F','Liquidation venue breadth percentile, 1h','F4Z__liquidation__venue_breadth_percentile__1h__value'),
('G','Funding persistence percentile, 5m','F4Z__funding__funding_persistence_percentile__5m__value'),
('H','Basis level robust z-score, 1h','F4Z__basis__basis_level_rz__1h__value')]
letter = {c:a for a,b,c in features}
new_contract = js(NEW/'CONTRACT.json')
assert set(new_contract['features']) == set(letter)

W,H = landscape(A4)
WIDTH = W-72
navy = colors.HexColor('#18384D')
teal = colors.HexColor('#007F86')
styles = {k:ParagraphStyle(k,fontName=font,fontSize=size,leading=lead,textColor=color,spaceAfter=space)
          for k,font,size,lead,color,space in [
          ('title','Helvetica-Bold',23,27,navy,9),('head','Helvetica-Bold',12,16,teal,7),
          ('body','Helvetica',10,14,navy,8),('small','Helvetica',8.5,11,navy,6),
          ('cell','Helvetica',8.5,10.5,navy,0),('th','Helvetica-Bold',8.5,10.5,colors.white,0),
          ('code','Courier',7.5,10,navy,5)]}

def build(hours):
    rows=[]; catalog={}; comparisons=[]
    def register(folder):
        code=f'S{len(catalog)+1}'
        catalog[code]=str(folder)
        if (folder/'CONTRACT.json').exists(): js(folder/'CONTRACT.json')
        if (folder/'COMPLETE.json').exists(): js(folder/'COMPLETE.json')
        return code
    def add(name,feat,recipe,row,src,protocol='Quarterly walk-forward'):
        rows.append(dict(model=name,features=feat,recipe=recipe,mae_log_bps=float(row.get('mae',row.get('mae_log_bps'))),
                         spearman_ic=float(row.get('rank_ic',row.get('spearman_ic',np.nan))),
                         rows=int(row['rows']),source=src,protocol=protocol))
    pair=E/f'two_feature_{hours}h_v1'; sid=register(pair)
    for _,r in csv(pair/'scores.csv').iterrows():
        name={'rf_baseline':'RF baseline','hybrid':'Hybrid - pair, tuned','mlp':'MLP - pair, tuned'}[r.model]
        add(name,'A B',{'rf_baseline':'R','hybrid':'H','mlp':'N'}[r.model],r,sid)
    additions=E/f'individual_{hours}h_additions_v1'; sid=register(additions)
    for _,r in csv(additions/'individual_results.csv').iterrows():
        feat='A B '+letter[r.added_feature]
        add(('Hybrid' if r.model=='hybrid' else 'MLP')+' - '+feat+', tuned',feat,'H' if r.model=='hybrid' else 'N',r,sid)
    if hours==4:
        folder=E/'four_feature_4h_v1';sid=register(folder)
        for _,r in csv(folder/'scores.csv').iterrows():
            if '_four_' in r.model:
                add(r.model.replace('_',' ').replace('mlp','MLP').replace('hybrid','Hybrid'), 'A B D F',
                    'H' if r.model.startswith('hybrid') else 'N',r,sid)
        for folder_name,suffixes in [('five_feature_4h_v1',['five']),('seven_eight_feature_4h_v1',['seven','eight'])]:
            folder=E/folder_name;sid=register(folder)
            contract=js(folder/'CONTRACT.json')
            for _,r in csv(folder/'scores.csv').iterrows():
                pack=r.model.split('_')[-1]
                if pack not in suffixes: continue
                feat=' '.join(sorted(letter[c] for c in contract['packs'][pack]))
                add(r.model.replace('_',' ').replace('mlp','MLP').replace('hybrid','Hybrid')+' - fixed recipe',
                    feat,'H' if r.model.startswith('hybrid') else 'N',r,sid)
        folder=E/'focused_four_feature_4h_v1';sid=register(folder)
        for _,r in csv(folder/'scores.csv').iterrows():
            if r.strategy=='selected':
                add(r.model.upper()+' - focused feature search','By quarter',('H' if r.model=='hybrid' else 'N')+' / S',r,sid)
        folder=E/'activation_4h_v1';sid=register(folder)
        for _,r in csv(folder/'scores.csv').iterrows():
            if r.activation=='silu': continue # identical to MLP A B F above
            add('MLP - '+r.activation.replace('_',' '),'A B F','N / A',r,sid)
    tree=W1/f'Experiments/tree_magnitude_filter_v1/{hours}h';sid=register(tree)
    tree_scores=csv(tree/'scores.csv')
    add('Filtered tree-model selection','By quarter','T',tree_scores.loc[tree_scores.split=='test'].iloc[0],sid)
    selections=js(tree/'selections.json')
    sid=register(NEW)
    for _,r in csv(NEW/'scores.csv').query('horizon_hours == @hours').iterrows():
        name={'median_regression':'Median regression - eight, frozen','mean_baseline':'Constant mean baseline',
              'median_baseline':'Constant median baseline'}[r.model]
        add(name,'A B C D E F G H' if r.model=='median_regression' else 'None','M' if r.model=='median_regression' else 'C',r,sid,'Frozen pre-2024')
    df=pd.DataFrame(rows)
    assert (df.rows==18481).all()
    df['mae_rank']=df.mae_log_bps.round(10).rank(method='min').astype(int)
    df['ic_rank']=df.spearman_ic.round(10).rank(method='min',ascending=False).astype('Int64')
    df=df.sort_values(['mae_rank','model']).reset_index(drop=True)
    df.to_csv(SUP/f'ScaleWave_{hours}h_MAE_IC_Ranked.csv',index=False)
    # Verify matching timestamps and target values, without recalculating model scores.
    reference=csv(NEW/f'predictions_{hours}h.csv')
    ref_index=pd.to_datetime(reference.iloc[:,0],utc=True)
    ref_truth=reference.truth_log_bps.to_numpy()
    for folder in [Path(p) for p in catalog.values() if Path(p)!=NEW]:
        path=folder/'predictions.csv'
        if path.exists():
            p=csv(path)
            if 'truth' not in p: raise ValueError(f'No truth column: {path}')
            groups=list(p.groupby('model',sort=False)) if 'model' in p else [('all',p)]
            for group,part in groups:
                idx=pd.to_datetime(part.iloc[:,0],utc=True)
                assert np.array_equal(idx,ref_index),(path,group)
                np.testing.assert_allclose(part.truth,ref_truth,rtol=0,atol=1e-8)
                comparisons.append(dict(path=str(path),group=group,rows=len(part),timestamp_and_target_match=True))
        else: raise FileNotFoundError(path)
    story=[]
    def p(t,k='body'):return Paragraph(t,styles[k])
    def addp(t,k='body'):story.append(p(t,k))
    def page(title,sub=''):
        if story:story.append(PageBreak())
        addp(f'SCALEWAVE  /  {hours}-HOUR MAGNITUDE  /  11 NOVEMBER 2026','small')
        addp(title,'title')
        if sub:addp(sub,'small')
    def table(headers,data,widths,compact=False):
        cells=[[p(escape(str(x)),'th') for x in headers]]+[[p(escape(str(x)),'cell') for x in r] for r in data]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#EEF4F6'),colors.white]),
                              ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
                              ('TOPPADDING',(0,0),(-1,-1),3 if compact else 5),('BOTTOMPADDING',(0,0),(-1,-1),3 if compact else 5)]))
        story.extend([t,Spacer(1,8)])
    chunk=19 if hours==1 else 18
    for start in range(0,len(df),chunk):
        page('Ranked model results' if start==0 else 'Ranked model results - continued',
             'MAE: lower is better. Spearman IC: higher is better. Ranks use full precision (ties within 1e-10).')
        if start==0:
            addp(f'Target: absolute BTC start-to-end log return over the next <b>{hours} hour(s)</b>. All rows below use the same <b>18,481 evaluation timestamps</b>; target and timestamp alignment were checked against saved predictions.','small')
        data=[]
        for _,r in df.iloc[start:start+chunk].iterrows():
            data.append([r.mae_rank,r.model,r.features,f'{r.mae_log_bps:.6f}', 'Undefined' if pd.isna(r.spearman_ic) else f'{r.spearman_ic:.6f}',
                         '-' if pd.isna(r.ic_rank) else int(r.ic_rank),r.recipe,r.source])
        table(['MAE rank','Model / tested procedure','Features','MAE log bps','Spearman IC','IC rank','Build','Source'],data,[43,290,115,89,85,45,55,WIDTH-722],compact=True)
        addp('A-H identify features on the feature page; Build codes identify construction recipes. Constant mean/median IC is undefined because every forecast is identical. Baselines are included in the MAE rank; undefined IC receives no rank.','small')
        if start+chunk>=len(df):
            addp('Ranking is descriptive of previously examined historical data. Frozen and quarterly refitted models have different training schedules. The ranking does not establish a fresh holdout winner or statistical significance.','small')
    page('Features used','Source features and derived price statistics; feature windows are not forecast horizons.')
    for a,b,c in features:
        addp(f'<b>{a} | {b}</b>','small');addp(escape(c),'code')
    addp('A, B, C, F, G and H are supplied F4Z inputs. D and E are derived from the separate canonical five-minute BTC return history. E = square root of the sum of squared five-minute log returns over 168 hours (2,016 returns), using endpoints strictly before the forecast. It extends beyond a one-day feature-window limit.','small')
    addp('D compares hourly-normalized realized volatility: (RV4h / sqrt(4)) / (RV24h / sqrt(24)), with a small denominator floor. The eight-feature list contains both supplied and derived inputs.','small')
    page('How the models were built','Build codes in the ranking connect each result to its construction recipe.')
    recipes=[
    ('C | Constant baselines','Calculate the training mean or median of the horizon-specific absolute target. Use that one number for every evaluation row. Fit once using labels ending before 1 January 2024. No input features; IC is undefined.'),
    ('M | Frozen median regression','All eight inputs; QuantileRegressor quantile 0.5, alpha 0, HiGHS solver. Training median imputation, mean/std scaling and one missingness indicator per feature. Fit separately for 1h/4h on 29,615 eligible pre-2024 rows; clip predictions at zero. No quarterly refit.'),
    ('R | RF baseline','Original pair A+B. RandomForestRegressor: 48 trees, absolute-error criterion, depth 3, minimum leaf 128, all features available at splits, bootstrap sample 4,096, seed 48. Training medians fill missing values. Refit quarterly on expanding history.'),
    ('H | Hybrid','Fit median regression (quantile 0.5, alpha 0, HiGHS) and RF on the same features. Original RF: 128 trees, absolute error, depth 3, leaf 128, bootstrap sample 4,096, max_features=1, seed 48. Prediction = (1-w) x median + w x RF; prior validation chooses w from 0, 0.1, 0.25, 0.5. Training median imputation, standardization and missingness flags.'),
    ('N | Residual MLP','Fit a training-only median-regression linear component (alpha 0.0001 on target/100), freeze it, then learn a nonlinear correction. Training median/IQR scaling with std fallback; values clipped to [-8,8]. SiLU hidden layers by default, zero-initialized output head, nonnegative output. Final forecasts average seeds 48,149,250.'),
    ]
    table(['Build','Construction'],recipes,[145,WIDTH-145])
    addp('MLP training: MAE objective, AdamW learning rate 0.001, weight decay 0.01 on nonlinear matrices, gradient clipping 1, batch size 512, at most 48 epochs. Prior validation selects epoch, patience 8, minimum improvement 0.01 bps, with epoch zero eligible. Outer refits use the chosen epoch counts.','small')
    page('Training and model selection','How configurations were selected and when the models were refitted.')
    addp('Earlier hybrid, MLP and tree experiments use four expanding outer windows starting 1 January, 1 April, 1 July and 1 October 2024. Training targets must end strictly before each fit cutoff; validation uses earlier outcomes only. Evaluation covers January 2024 into January 2025.','small')
    table(['Experiment','Features and fitting choices'],[
        ('Pair and single additions','A+B, or A+B plus one of C-H. Each addition is retuned separately. Hybrid compares the original RF recipe against 96 trees / depth 4 / leaf 128 / sample 2,048 / squared error. MLP compares [8,4] units with dropout 0.05 against [4] units with dropout 0.15. Prior-quarter MAE chooses configuration; MLP selection screens seed 48 then fits the chosen configuration with three seeds.'),
        ('Frozen eight-feature model','The new median regression and constant baselines train once before 2024. Targets are rebuilt from 12 consecutive five-minute returns for 1h, or 48 for 4h. Existing 48h labels are not reused.'),
        ('T | Filtered tree selection','Training-only quality, association, mutual-information/null, stability and redundancy screening; then raw/derived feature comparisons and drop ablations. Two preceding quarterly validation blocks choose model/features. Candidates: LightGBM stumps or depth 2, CatBoost depth 2, RF depth 3. One selected model per outer quarter is refitted and scored; the pooled result is a selection procedure, not one fixed model.'),
    ],[150,WIDTH-150])
    addp('Tree settings: LightGBM uses MAE objective, 48 stumps or 64 depth-2 trees, learning rate 0.03, minimum child count 512, L2=50, L1=5, max_bin=31. CatBoost uses MAE, 80 trees, depth 2, learning rate 0.03, L2=50, no bootstrap. RF uses the R recipe above. Seed 48; training median imputation; nonnegative predictions.','small')
    if hours==4:
        addp('Additional 4h experiments','head')
        addp('Fixed sets: four = A B D F; five = A B D E F; seven = A B D E F G H; eight = A-H. Fixed recipes retain the original hybrid RF with validation-selected blend, or [8,4] SiLU MLP/dropout 0.05 with validation-selected epochs.','small')
        addp('Four-feature tuning compares original/regularized/squared-error RF recipes and MLP widths [8,4], [4], [16,8], [8]. A change needs validation MAE gain of at least 0.1 bps and gains in at least two months. No MLP change passed, so its selected/default scores tie.','small')
        page('Additional 4h tests','Feature-combination and activation comparisons are separate procedures.')
        addp('S | Focused four-feature search','head')
        addp('Keep A+B and screen all 15 pairs from C-H using up to 6,144 training rows. Reduced screen: 24-tree hybrid or 16-epoch, one-seed MLP. Fully validate the top three plus anchor A B D F. Refit only the validation winner on full outer training history. Thus the 15 screen candidates do not each have pooled outer-test scores.','small')
        table(['Quarter','Selected hybrid features','Selected MLP features'],[
              ['January 2024','A B E F','A B E F'],['April 2024','A B D F','A B D F'],
              ['July 2024','A B D E','A B D E'],['October 2024','A B F H','A B D F']],[140,300,WIDTH-440])
        addp('A | Activation study','head')
        addp('Use A B F and change only the nonlinear activation: SiLU, ReLU, Leaky ReLU (slope 0.01), GELU, tanh. Widths/dropout remain the earlier selected [8,4]/0.05 in January and July, [4]/0.15 in April and October. Same seeds and optimizer; validation still chooses epochs. A separate strategy picks the activation by prior-validation ensemble MAE each quarter.','small')
        addp('The SiLU row is the same saved result as MLP A B F in the single-addition experiment, so it is listed once. Repeated baseline/control rows copied across experiment score files are also listed once. The four-feature selected/default MLP strategies remain separately labeled because both procedures were reported.','small')
    page('Tree-selection features by quarter','These inputs differ from the fixed eight-feature set. Only actually selected columns are listed.')
    extra=sorted({c for s in selections for c in s['columns'] if c not in letter})
    aliases={c:f'T{i+1}' for i,c in enumerate(extra)}
    tree_rows=[]
    for s in selections:
        tree_rows.append([s['issue'][:10],s['model'],', '.join(letter.get(c,aliases.get(c)) for c in s['columns'])])
    table(['Fit date','Selected model','Selected inputs'],tree_rows,[110,130,WIDTH-240])
    for col,alias in aliases.items():
        addp(f'<b>{alias}</b> | '+escape(col).replace('___','___<wbr/>').replace('__','__<wbr/>'),'small')
    addp('PRODUCT__x___y = tanh(x/3) x tanh(y/3). MEAN24__x is the prior 24-hour rolling mean (current timestamp excluded). SLOG__x = sign(x) x log(1 + abs(x)). A/B and other shared letters refer to the main feature page. Exact selected columns and model names are exported in the supporting JSON.','small')
    page('Sources and scope','Saved score files and contracts were inspected; no training or new forecasting experiment was run.')
    data=[]
    for sid,path in catalog.items():
        path=Path(path)
        label=('Week 2/' + str(path.relative_to(ROOT)).replace('\\','/')) if path.is_relative_to(ROOT) else ('Week 1/'+str(path.relative_to(W1)).replace('\\','/'))
        data.append([sid,label])
    table(['Source','Run folder (score CSV, contract, completion and prediction records)'],data,[50,WIDTH-50])
    addp('Metrics are taken from saved pooled evaluation score tables, not training or validation tables. Alignment verification reads saved prediction timestamps/targets; scores were not independently recomputed or model checkpoints replayed for this report. Per-row CSV sources and hashes are saved in the supporting manifest.','small')
    addp('Not included: 48h conditional UP/DOWN experiments; partial-window scores; training/validation-only trial scores. Folders rf_fixed_liquidation_1h_v1 and mlp_fixed_liquidation_1h_v1 concern a 1h feature within a 48h forecast and are excluded. Scope is the completed short-horizon runs identified in the inspected Week 1/Week 2 experiment folders.','small')
    addp('MAE is measured in log basis points, not ordinary percentage points. IC means Spearman rank correlation, not directional accuracy. Undefined constant-baseline IC is deliberately not replaced with zero.','small')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(navy)
        canvas.drawString(36,18,f'ScaleWave | {hours}h magnitude | Historical results')
        canvas.drawRightString(W-36,18,str(doc.page))
    target=OUT/f'ScaleWave_{hours}h_Models_Ranked_MAE_IC_2026-11-11.pdf'
    SimpleDocTemplate(str(target),pagesize=(W,H),leftMargin=36,rightMargin=36,topMargin=30,bottomMargin=30).build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(target)
    text='\n'.join(p.extract_text() for p in reader.pages)
    for _,r in df.iterrows():assert f'{r.mae_log_bps:.6f}' in text
    (SUP/f'{hours}h_tree_selections.json').write_text(json.dumps(selections,indent=2),encoding='utf-8')
    return dict(path=str(target),pages=len(reader.pages),ranked_rows=len(df),alignment_checks=comparisons)

results=[build(h) for h in [1,4]]
for path in [W1/'Code'/n for n in ['canonical_return_features.py','tree_magnitude_filter.py','search_report_features.py','individual_hybrid_additions.py','four_feature_4h.py','eight_feature_search.py','linear_initialized_mlp.py']]:track(path)
for kind in ['pair','rf']:track(W1/f'Experiments/nested_boost_48h_v1/models/2024-01__report_outer_{kind}__expanding.joblib')
track(E/'focused_four_feature_4h_v1/SELECTIONS.json')
track(Path(__file__))
(SUP/'manifest.json').write_text(json.dumps(dict(reports=results,source_sha256=sources),indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
