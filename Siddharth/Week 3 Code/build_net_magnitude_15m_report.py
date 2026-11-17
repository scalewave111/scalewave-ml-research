"""Build an audited PDF from saved forecasts/model metadata; no fitting/inference."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import sys
from xml.sax.saxutils import escape
import joblib
import numpy as np
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak, KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from pypdf import PdfReader
from fit_net_magnitude_48h import GRURegressor

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'Code/experiments'; OUT=ROOT/'Report'
SUP=OUT/'net_magnitude_15m_report_support_2026-11-16'
PDF=OUT/'BTC_15m_Net_Magnitude_Models_Features_Rankings_2026-11-16.pdf'
RUNS={'hgb':'20261116T055443612541Z','extra':'20261116T055943348175Z','mlp':'20261116T061517849147Z','gru':'20261116T061932445544Z'}
LABEL={'hgb':'HGB','extra':'ExtraTrees','mlp':'MLP','gru':'GRU'}
SUP.mkdir(parents=True,exist_ok=True)
hashes={}; data={}; contracts={}; cards={}; picks={}; metrics_rows=[]; ledger=[]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def track(p):hashes[str(p)]=sha(p);return p
def js(p):return json.loads(track(p).read_text(encoding='utf-8-sig'))
def csv(p):return pd.read_csv(track(p))
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,default=str,allow_nan=False),encoding='utf-8')
def metric(g,col):
    e=g[col].to_numpy(float)-g.actual.to_numpy(float)
    return dict(rows=len(g),mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.square(e).mean())),bias=float(e.mean()))

reference=None
for family,stamp in RUNS.items():
    path=EXP/f'net_magnitude_15m_{family}_only_v1'/stamp
    done=js(path/'COMPLETE.json'); assert done['status']=='COMPLETE'
    f=csv(path/'predictions.csv'); assert sha(path/'predictions.csv')==done['prediction_sha256']
    f.decision_time=pd.to_datetime(f.decision_time,utc=True);f.issue=pd.to_datetime(f.issue,utc=True)
    f=f.sort_values('decision_time').reset_index(drop=True)
    assert len(f)==18480 and f.decision_time.is_unique and np.isfinite(f[['actual',family,'mean','median']]).all().all()
    if reference is None:reference=f[['decision_time','actual','mean','median']]
    else:pd.testing.assert_frame_equal(reference,f[['decision_time','actual','mean','median']])
    data[family]=f; contracts[family]=js(path/'CONTRACT.json'); picks[family]=js(path/'selections.json'); cards[family]=[]
    saved=csv(path/'metrics.csv')
    for period,g in [('ALL',f)]+[(str(k),v) for k,v in f.groupby('issue')]:
        values=metric(g,family); metrics_rows.append(dict(family=family,model=LABEL[family],period=period,**values))
    for choice in picks[family]:
        modelpath=path/choice['artifact'];assert sha(track(modelpath))==choice['sha256']
        a=joblib.load(modelpath); assert a['prep']['cols']==choice['features']
        assert a['max_label_end']<pd.Timestamp(choice['issue'])
        model=a['model'];record={**choice,'transformed_inputs':2*len(choice['features']),'target_center':a['center'],'target_scale':a['scale'],'estimator_rows':a['estimator_rows']}
        if family=='gru':
            record.update(inputs=model.inputs,state_shapes={k:list(v.shape) for k,v in model.state.items()},parameters=sum(v.numel() for v in model.state.values()))
        else:
            record['parameters']=model.get_params(deep=False)
            if family=='mlp':record.update(layer_shapes=[list(w.shape) for w in model.coefs_],epochs=model.n_iter_,train_loss=float(model.loss_),parameter_count=sum(w.size for w in model.coefs_)+sum(w.size for w in model.intercepts_))
            if family=='hgb':record['iterations']=model.n_iter_
            if family=='extra':record['trees']=len(model.estimators_)
        cards[family].append(record)
    trials=csv(path/'validation_trials.csv')
    assert len(trials)==60
    for candidate,g in trials.groupby(['issue','candidate'],sort=False):
        spec=ast.literal_eval(g.spec.iloc[0]); assert spec['model']==family
        pooled=float(np.average(g.mae,weights=g.rows))
        ledger.append(dict(family=family,issue=candidate[0],candidate=int(candidate[1]),pack=spec['pack'],validation_mae=pooled))
    for choice in picks[family]:
        g=[v for v in ledger if v['family']==family and v['issue']==choice['issue']]
        selected=next(v for v in g if v['candidate']==choice['candidate'])
        assert np.isclose(selected['validation_mae'],choice['validation_mae'])
        assert selected['validation_mae']<=min(v['validation_mae'] for v in g)+1e-8
    shutil.copyfile(path/'validation_trials.csv',SUP/f'{family}_validation_trials.csv')
    shutil.copyfile(path/'CONTRACT.json',SUP/f'{family}_contract.json')
    track(path/'runner_snapshot.py')
    m=metric(f,family);s=saved[(saved.period=='ALL')&(saved.model==family)].iloc[0]
    assert abs(m['mae']-s.mae)<1e-5 and abs(m['rmse']-s.rmse)<1e-5

metrics=pd.DataFrame(metrics_rows)
ranking=metrics[metrics.period=='ALL'].copy()
for col in ['mean','median']:
    ranking=pd.concat([ranking,pd.DataFrame([dict(family=col,model=col.title()+' baseline',period='ALL',**metric(reference,col))])],ignore_index=True)
ranking['mae_rank']=ranking.mae.rank(method='min').astype(int)
ranking['rmse_rank']=ranking.rmse.rank(method='min').astype(int)
ranking=ranking.sort_values('mae');ranking.to_csv(SUP/'model_rankings.csv',index=False)
metrics.to_csv(SUP/'period_metrics.csv',index=False)
pd.DataFrame(ledger).to_csv(SUP/'feature_pack_validation_scores.csv',index=False)
dump(SUP/'model_metadata.json',cards)
formulas=csv(EXP/f'net_magnitude_15m_hgb_only_v1/{RUNS["hgb"]}/new_feature_formulas.csv')
assert len(formulas)==57
formulas.to_csv(SUP/'new_feature_formulas.csv',index=False)
used=sorted(set(c for fam in picks.values() for p in fam for c in p['features']))
membership=[]
for feat in used:
    row=dict(feature=feat)
    for family in RUNS:
        row[family]=','.join(pd.Timestamp(p['issue']).strftime('%b') for p in picks[family] if feat in p['features']) or '-'
    membership.append(row)
pd.DataFrame(membership).to_csv(SUP/'exact_feature_membership.csv',index=False)
dump(SUP/'candidate_feature_packs.json',contracts['hgb']['packs'])

# Readable pages: 12-point prose and 10.5/11-point exact feature identifiers.
NAVY=colors.HexColor('#17354A');TEAL=colors.HexColor('#087E8B');LIGHT=colors.HexColor('#EDF4F7')
styles={
    'title':ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=26,leading=31,textColor=NAVY,spaceAfter=14),
    'h':ParagraphStyle('h',fontName='Helvetica-Bold',fontSize=19,leading=23,textColor=NAVY,spaceAfter=12),
    'sub':ParagraphStyle('sub',fontName='Helvetica-Bold',fontSize=13,leading=17,textColor=TEAL,spaceBefore=8,spaceAfter=6),
    'body':ParagraphStyle('body',fontName='Helvetica',fontSize=12,leading=16,spaceAfter=9),
    'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=11,leading=14,splitLongWords=True),
    'small':ParagraphStyle('small',fontName='Helvetica',fontSize=10.5,leading=13,splitLongWords=True),
}
story=[]
def para(t,kind='body'):return Paragraph(escape(str(t)).replace('\n','<br/>'),styles[kind])
def p(t,kind='body'):story.append(para(t,kind))
def page(title):
    if story:story.append(PageBreak())
    p(title,'h')
def table(headers,rows,widths,small=False):
    body=[[para(c,'small' if small else 'cell') for c in headers]]+[[para(c,'small' if small else 'cell') for c in row] for row in rows]
    t=Table(body,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),colors.white),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),1,TEAL)]))
    # Paragraph text colour must be set explicitly inside dark header cells.
    for j,h in enumerate(headers):body[0][j]=Paragraph('<font color="white">'+escape(str(h))+'</font>',styles['cell'])
    t._cellvalues[0]=body[0]
    story.append(t)

page('15-minute net magnitude | HGB selected')
p('BTCUSDT - models, features, formulas and rankings','title')
p('16 November 2026 | Four independent model families | 18,480 matched observations')
p('Target: net_magnitude_log_bps = 10,000 x abs[ln(P(t+15min)/P(t))]. It measures absolute endpoint movement, rather than UP minus DOWN excursion.')
p('Selected model: HGB, following the user selection. Its pooled MAE is 11.682119 log bps, 10.05% below median and 11.51% below mean. ExtraTrees has the lowest RMSE. Every model is evaluated separately across the same timestamps.')
table(['MAE rank','Model / baseline','MAE','RMSE','RMSE rank'],[[r.mae_rank,r.model,f'{r.mae:.6f}',f'{r.rmse:.6f}',r.rmse_rank] for r in ranking.itertuples()],[65,235,155,155,145])
p('Lower scores are better. HGB selection follows this historically exposed comparison; it is not independent test confirmation. No model was refitted for this report.','small')

page('Selected HGB | quarterly models')
hgb=data['hgb']; selected_score=metric(hgb,'hgb')
hgbnames=sorted(set(n for a in cards['hgb'] for n in a['features']))
p(f'HGB is the selected family for every period. It has {len(hgbnames)} distinct base features across its four saved quarterly estimators; exact input order remains in model_metadata.json. Each quarterly feature pack was chosen using earlier validation.')
rows=[]
for a in cards['hgb']:
    g=hgb[hgb.issue==pd.Timestamp(a['issue'])];m=metric(g,'hgb');b=metric(g,'median')
    rows.append([pd.Timestamp(a['issue']).strftime('%b %Y'),a['spec']['pack'],len(a['features']),a['transformed_inputs'],f'{m["mae"]:.5f}',f'{m["rmse"]:.5f}','PASS' if a['accepted'] else 'FAIL'])
table(['Issue','Feature pack','Base d','Inputs 2d','MAE','RMSE','Gate'],rows,[90,145,70,90,130,130,100])
p('Architecture: initial constant + 60 boosted shallow regression trees; maximum depth 3 and 7 leaves per tree. Loss: absolute error. Input is a median/IQR-scaled feature vector with a missing flag for every base feature. Output is transformed back to log bps and clipped at zero.')
p('Hyperparameters: learning_rate=0.1; max_iter=60; max_leaf_nodes=7; max_depth=3; min_samples_leaf=64; l2_regularization=5; max_bins=255; early_stopping=False; random_state=48. Saved models confirm 60 fitted iterations in all four periods.')
p('These are four issue-specific HGB estimators of one model family. There is no GRU/MLP/ExtraTrees substitution and no median fallback in the primary HGB forecast.')
dump(SUP/'selected_hgb.json',dict(selected_by='user',target='net_magnitude_log_bps',horizon_minutes=15,family='hgb',run=str(EXP/f'net_magnitude_15m_hgb_only_v1/{RUNS["hgb"]}'),scores=selected_score,models=cards['hgb'],selection_limit='Family chosen after historically exposed comparison; no refitting or deployment.'))

periodrows=[]
for issue,g0 in data['hgb'].groupby('issue'):
    for fam in RUNS:
        g=data[fam][data[fam].issue==issue];m=metric(g,fam)
        periodrows.append([issue.strftime('%b %Y'),LABEL[fam],len(g),f'{m["mae"]:.5f}',f'{m["rmse"]:.5f}'])
    for baseline in ['mean','median']:
        m=metric(g0,baseline);periodrows.append([issue.strftime('%b %Y'),baseline.title(),len(g0),f'{m["mae"]:.5f}',f'{m["rmse"]:.5f}'])
for start in [0,12]:
    page('All models | period scores'+(' continued' if start else ''))
    p('The October issue includes the final January 2025 evaluation observations. All errors are in log bps.','small')
    table(['Issue','Model','Rows','MAE','RMSE'],periodrows[start:start+12],[130,180,95,175,175])

page('Population, validation and feature selection')
audit=contracts['hgb']['input_audit']
p(f'Canonical BTCUSDT horizon_minutes=15 labels are read from the original package. Original X grid: {audit["original_rows"]:,} rows; eligible labels: {audit["eligible_rows"]:,}; evaluation: {audit["evaluation_rows"]:,}. One source-binding crossing origin is excluded. Evaluation extends from {reference.decision_time.min()} to {reference.decision_time.max()}.')
p('The original forecast grid is every 30 minutes. This is a 15-minute target predicted at those origins, not a full minute-by-minute or 15-minute X population. Invalid labels are excluded and never imputed. The feature timeline is built before label filtering, preserving sequence availability.')
p('For each Jan/Apr/Jul/Oct 2024 issue, five feature packs are compared in three earlier two-month windows, starting 9/6/3 months before issue. Training labels must mature strictly before each fit cutoff: origin+15min < cutoff. Inner validation label_end must precede the outer issue. Pack ranking uses row-count-weighted pooled validation MAE.')
p('Each family stays fixed across all periods. The primary forecast always uses the validation-best pack for that family. The audit gate requires >=0.1% improvement versus median MAE, >=2/3 window wins or ties and no >2% window MAE loss. All 16 saved quarterly choices pass this gate.')
p('Feature admission uses only mature training rows: >=80% finite coverage and more than one distinct value. No old feature has a reserved place. This is a feature-group comparison, not exhaustive individual forward selection or proof of marginal feature importance.')

page('Preprocessing and prediction equations')
p('For each admitted feature j, training median m_j and IQR q_j=max(P75-P25,1e-6) are calculated on all mature rows. Missing values are imputed by m_j. The scaled input is z_j=clip((x_j-m_j)/q_j,-8,8); a_j=1 if x_j is missing, otherwise 0. Input vector concatenates all z values and all a flags; d base features become 2d model inputs.')
p('Target normalization: c=median(y_train); s=max(sample_std(y_train),1); u=(y-c)/s. Every model fits u. Final prediction is max(0,c+s*f(z,a)). HGB uses absolute-error loss; ExtraTrees and MLP use squared error; GRU uses absolute error.')
p('At most 6,000 evenly spaced mature training rows fit the estimator. Feature quality, preprocessing, target scaling and mean/median baselines use all mature training rows. Seed=48. GRU is CPU float32 with exact-time snapshots at 90/60/30/0 minutes; there is no nearest-time filling.')
rows=[]
for a in cards['hgb']:
    rows.append([pd.Timestamp(a['issue']).strftime('%b'),a['training_rows'],a['estimator_rows'],f'{a["target_center"]:.7f}',f'{a["target_scale"]:.7f}',a['max_label_end']])
table(['Issue','Mature rows','Fit rows','Target c','Target s','Latest mature label end'],rows,[60,95,85,120,120,275])
p('A feature recorded at decision time may itself use an inherited availability mask. Upstream F4Z generation and normalization are inherited from the supplied data; this report does not independently reconstruct their generating distributions.','small')

descriptions={
'hgb':('HGB','Additive boosted regression trees; scalar output.','Absolute-error loss; 60 iterations; learning_rate0.1; depth3; max7leaves; min_leaf64; L2=5; max_bins255; early_stoppingFalse; seed48.'),
'extra':('ExtraTrees','Average of 48 randomized regression trees.','Squared-error splits; 48trees; depth10; min_leaf32; min_split2; max_features0.8; bootstrapFalse; n_jobs1; seed48.'),
'mlp':('MLP','2d inputs -> Dense16/ReLU -> Dense1/identity.','Squared-error regression; Adam; alpha1; learning_rate0.001; batch256; 40epochs maximum; no early stopping; shuffleFalse; beta1=.9,beta2=.999,epsilon1e-8; tol1e-4; n_iter_no_change10; seed48.'),
'gru':('GRU','4 snapshots x 2d inputs -> one-layer GRU16 -> final hidden state -> Linear16-to1.','L1 loss; 24epochs; batch256; AdamW lr0.001,decay0.03,betas(.9,.999),eps1e-8; gradient clip1; no dropout/bidirectionality/shuffle; CPUfloat32; seed48.')}
for fam in RUNS:
    page(LABEL[fam]+' | architecture and hyperparameters')
    title,architecture,hyper=descriptions[fam];p(architecture);p(hyper)
    if fam=='hgb':p('Normalized output: initial constant + learning_rate x sum(tree_leaf_output). A single HGB family uses a separate earlier-validation pack choice for each issue.')
    if fam=='extra':p('Normalized output: arithmetic mean of the fitted tree outputs. Each tree is trained within the same family and fitting cutoff.')
    if fam=='mlp':p('Normalized output: W2*ReLU(W1*input+b1)+b2. An iteration-limit warning does not invalidate saved predictions; convergence was not guaranteed within the fixed budget.')
    if fam=='gru':p('Gate equations: r=sigmoid(W_ir*x+b_ir+W_hr*h_prev+b_hr); z=sigmoid(W_iz*x+b_iz+W_hz*h_prev+b_hz); n=tanh(W_in*x+b_in+r*(W_hn*h_prev+b_hn)); h=(1-z)*n+z*h_prev. Output=W_out*h_last+b_out. Gates use elementwise multiplication.','small')
    rows=[]
    for a in cards[fam]:
        fitted=(str(a['iterations'])+' rounds' if fam=='hgb' else str(a['trees'])+' trees' if fam=='extra' else f'{a["epochs"]} epochs / {a["parameter_count"]} params' if fam=='mlp' else f'{a["parameters"]} params')
        rows.append([pd.Timestamp(a['issue']).strftime('%b'),a['spec']['pack'],len(a['features']),a['transformed_inputs'],fitted])
    table(['Issue','Feature pack','Base d','Inputs 2d','Fitted architecture'],rows,[75,165,85,110,320])
    p('All saved estimator parameters and GRU tensor shapes are exported in model_metadata.json. Exact fitted feature order is included for every model.','small')

page('Five candidate packs and quarterly choices')
table(['Pack','Candidates','Contents'],[
['market',522,'Original masked F4Z value candidates. Only training-eligible columns enter each fit.'],
['prior_price',51,'Existing 15m UP/DOWN price features: 8 measures x6 lookbacks plus3volatility ratios.'],
['new_price',57,'Nine measures x6 short/long lookbacks plus3normalized volatility ratios.'],
['all',630,'Union market522 + prior_price51 + new_price57.'],
['market_new',579,'Union market522 + new_price57.']],[140,95,520])
p(f'The four primary families use {len(used)} distinct base features in total; HGB uses {len(hgbnames)}. Candidate pack size is different from actual fitted feature count because admission is training-only. Exact candidate and fitted ordered lists are in the support files.')
choices=[]
for fam in RUNS:
    for a in cards[fam]:choices.append([LABEL[fam],pd.Timestamp(a['issue']).strftime('%b'),a['spec']['pack'],f'{a["validation_mae"]:.5f}',len(a['features'])])
for start in [0,8]:
    page('Quarterly feature-pack choices'+(' | neural models' if start else ' | tree models'))
    table(['Model','Issue','Pack','Earlier validation MAE','Fitted base features'],choices[start:start+8],[100,80,180,205,190])

page('New features | formula definitions')
p('Let r_i=10,000*ln(P_i/P_(i-5min)), indexed at its ending time. For a decision t and lookback m minutes, use endpoints t-m <= i < t. Require all m/5 finite 5-minute returns. No endpoint at t or in the future enters a new feature.')
p('Define N=sum(r), A=sum(abs(r)), Q=sum(r^2), U=sum(max(r,0)^2), J=max(abs(r)). Six lookbacks m={15,30,60,240,1440,10080} minutes give 54 features. Three ratios give a total of57 new features.')
table(['Measurement','Formula'],[['RV','sqrt(Q)'],['Signed / absolute net','N / abs(N) (two separate features)'],['Absolute movement sum','A'],['Trend efficiency','abs(N)/max(A,1e-12)'],['Upside variance share','U/max(Q,1e-12)'],['Jump concentration','J^2/max(Q,1e-12)'],['RMS return','sqrt(Q/(m/5))'],['Positive fraction','mean(1[r>0])']],[240,515])
p('Normalized volatility ratios: (RV_short/sqrt(short))/max(RV_long/sqrt(long),1e-9), for minute pairs15/60,30/240,60/1440. RMS and RV at a fixed lookback differ only by a constant scale.','small')
for start in range(0,len(formulas),9):
    page(f'Exact new-feature formulas | {start+1}-{min(start+9,len(formulas))}')
    table(['Exact feature name','Implemented formula'],[[r.feature,r.formula] for r in formulas.iloc[start:start+9].itertuples()],[280,475],True)

page('Inherited price features | formulas')
p('The prior_price pack contains51 price features from Week2 improve_excursion_features_15m.py. It is distinct from the 57 newly created NET15 features. Every inherited rolling return window excludes its decision-time endpoint and requires complete 5m coverage.')
p('For h={1,4,12,24,48,168} hours, n=12h: RV=sqrt(sum r^2); up_RV=sqrt(sum max(r,0)^2); down_RV=sqrt(sum min(r,0)^2); net=sum r; max_abs_return=max abs(r).')
p('Let L=cumsum(fillna(r,0)).shift(1). Compute high=max(L) and low=min(L) across n+1 grid points. past_range=high-low; distance_from_high=high-L; distance_from_low=L-low. Values are masked unless the past window has n finite returns. These path measures use 5m close-to-close levels, not intrabar highs/lows.')
p('Three inherited volatility ratios use (RV_short/sqrt(short))/max(RV_long/sqrt(long),1e-6), with hour pairs1/24,4/24,24/168. Six lookbacks x8measurements plus3ratios =51. Exact identifiers and issue membership appear in the following inventory.')
p('F4Z names encode the family, measurement, lookback and value field. Historical active rate is a frequency, not current liquidation size. Flow activity is not signed buy-minus-sell flow. RZ and percentile are distinct normalizations. Their upstream reference formulas remain inherited, not independently reconstructed.')

# One comprehensive matrix includes HGB and every comparison family, without
# duplicating hundreds of identifiers in an additional inventory.
ordered=sorted(membership,key=lambda r:(r['feature'] not in hgbnames,r['feature']))
for start in range(0,len(ordered),8):
    page(f'Exact fitted feature membership | {start+1}-{min(start+8,len(ordered))}')
    p('Jan/Apr/Jul/Oct identify issue membership. HGB-used features come first. Each base feature has an additional missing indicator; GRU repeats its inputs at90/60/30/0minutes. Exact fitted order is preserved in support.','small')
    rows=[[r['feature'],r['hgb'],r['extra'],r['mlp'],r['gru']] for r in ordered[start:start+8]]
    table(['Exact base feature','HGB','ExtraTrees','MLP','GRU'],rows,[400,90,90,90,85],True)

page('Audit-policy scores and interpretation')
p('Primary rankings use the independent pure-family forecasts. The separately saved guarded_selected forecast may use the median on a failed gate. Here all16quarterly choices pass; these audit policies therefore match the corresponding primary forecasts.')
table(['Audit policy','MAE','RMSE'],[[LABEL[fam]+' guarded',f'{metric(f,"guarded_selected")["mae"]:.6f}',f'{metric(f,"guarded_selected")["rmse"]:.6f}'] for fam,f in data.items()],[335,210,210])
p('Selecting HGB optimizes the observed pooled MAE objective, while ExtraTrees is better for RMSE. A new feature being present in a chosen pack does not establish its individual causal value or marginal error reduction. No new inference or fitting was used to make this report.')

page('Provenance and reproducibility')
p('Verified four COMPLETE records and prediction SHA256s, plus all16saved-model SHA256s. Aligned18,480 unique origins, actual targets, mean and median baselines across all families. Recomputed pooled and period MAE/RMSE/bias and matched saved pooled scores within1e-5. Inspected fitted sklearn parameters and GRU tensor metadata.')
p('Audited240validation-trial rows and verified that each saved pack has minimum pooled earlier-validation MAE within its family. Original runs reported successful saved-model replay; replay was not rerun here. No model training or prediction was performed for report creation.')
p('Historical2022-2025 data have already informed research choices. All gates passed on earlier validation, but this is not a fresh untouched-test result or statistical significance claim. Inherited masked feature availability and upstream normalization limits remain.')
for fam,stamp in RUNS.items():p(f'{LABEL[fam]}: Code/experiments/net_magnitude_15m_{fam}_only_v1/{stamp}/','small')
p('Support includes selected_hgb.json, exact ordered fitted metadata, rankings, period metrics, feature membership,57new formulas, all candidate packs, contracts,240validation rows and hash audit. Builder: Code/build_net_magnitude_15m_report.py.','small')

def footer(canvas,doc):
    canvas.setStrokeColor(TEAL);canvas.line(40,34,802,34)
    canvas.setFont('Helvetica',9);canvas.setFillColor(NAVY)
    canvas.drawString(40,21,'BTCUSDT | 15m net magnitude | Selected HGB | 16 Nov 2026 | log bps')
    canvas.drawRightString(802,21,str(doc.page))

doc=SimpleDocTemplate(str(PDF),pagesize=landscape(A4),leftMargin=43,rightMargin=43,topMargin=32,bottomMargin=48,title='BTC 15m Net Magnitude - Selected HGB, Models and Features',author='ScaleWave research')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(str(PDF));text='\n'.join(page.extract_text() for page in reader.pages)
assert all(v in text for v in ['11.682119','ExtraTrees','NET15__rv_ratio_60_1440','GRU'])
assert all(feat in text.replace('\n','') for feat in used)
dump(SUP/'audit.json',dict(pdf=str(PDF),pages=len(reader.pages),prediction_runs_verified=4,model_hashes_verified=16,observations=18480,distinct_used_base_features=len(used),selected_hgb_feature_union=len(hgbnames),new_features=57,hashes=hashes,pdf_sha256=sha(PDF),training_performed=False,inference_performed=False))
print(json.dumps(dict(pdf=str(PDF),pages=len(reader.pages),support=str(SUP),hgb_features=len(hgbnames),all_used_features=len(used))))


