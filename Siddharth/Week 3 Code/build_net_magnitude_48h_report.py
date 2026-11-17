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
SUP=OUT/'net_magnitude_48h_report_support_2026-11-16'
PDF=OUT/'BTC_48h_Net_Magnitude_Models_Features_Rankings_2026-11-16.pdf'
RUNS={'hgb':'20261116T045400809455Z','extra':'20261116T045523854253Z','mlp':'20261116T045802302202Z','gru':'20261116T045940264860Z'}
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
    path=EXP/f'net_magnitude_48h_{family}_only_v1'/stamp
    done=js(path/'COMPLETE.json'); assert done['status']=='COMPLETE'
    f=csv(path/'predictions.csv'); assert sha(path/'predictions.csv')==done['prediction_sha256']
    f.decision_time=pd.to_datetime(f.decision_time,utc=True);f.issue=pd.to_datetime(f.issue,utc=True)
    f=f.sort_values('decision_time').reset_index(drop=True)
    assert len(f)==18385 and f.decision_time.is_unique and np.isfinite(f[['actual',family,'mean','median']]).all().all()
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
formulas=csv(EXP/f'net_magnitude_48h_hgb_only_v1/{RUNS["hgb"]}/new_feature_formulas.csv')
assert len(formulas)==48
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

page('48-hour net magnitude | model report')
p('BTCUSDT - architectures, feature formulas and model rankings','title')
p('16 November 2026 | Four independent model families | 18,385 matched observations')
p('Target: net_magnitude_log_bps = 10,000 x abs[ln(P(t+48h) / P(t))]. This is the absolute endpoint return; UP minus DOWN excursions is a different quantity.')
p('GRU has the lowest observed MAE: 181.870331 log bps. HGB is second among learned models at 182.909410. GRU improves MAE by 1.34% versus the mean baseline and 1.87% versus the median baseline. The mean baseline retains the lowest RMSE.')
table(['MAE rank','Model / baseline','MAE','RMSE','RMSE rank'],[[r.mae_rank,r.model,f'{r.mae:.6f}',f'{r.rmse:.6f}',r.rmse_rank] for r in ranking.itertuples()],[65,235,155,155,145])
p('Ranking uses full-period pooled MAE, lower is better. Each learned family runs in every period without another family or a baseline replacing it. Earlier validation selects its feature pack. These historical evaluation results do not imply all validation gates passed.','small')

page('Scores by evaluation period')
p(f'Coverage: {reference.decision_time.min()} to {reference.decision_time.max()}. The October issue continues through the final January 2025 observations.')
periodrows=[]
for issue,g0 in data['hgb'].groupby('issue'):
    label=issue.strftime('%b %Y')
    for fam in RUNS:
        g=data[fam][data[fam].issue==issue];m=metric(g,fam)
        periodrows.append([label,LABEL[fam],len(g),f'{m["mae"]:.4f}',f'{m["rmse"]:.4f}'])
    for baseline in ['mean','median']:
        m=metric(g0,baseline);periodrows.append([label,baseline.title(),len(g0),f'{m["mae"]:.4f}',f'{m["rmse"]:.4f}'])
for start in [0,12]:
    if start:page('Scores by evaluation period | continued')
    table(['Issue','Model','Rows','MAE (log bps)','RMSE (log bps)'],periodrows[start:start+12],[130,180,95,175,175])

page('Training, feature selection and prediction rules')
p('The canonical label is read directly from the checksum-verified October package. Valid finite nonnegative labels are retained; 96 source-binding crossing origins are excluded. The original 48,096-row grid becomes 48,000 eligible rows, with 18,385 evaluation origins.')
p('At each Jan/Apr/Jul/Oct 2024 issue, every feature pack is evaluated in three earlier two-month windows starting 9, 6 and 3 months before the issue. Training label_end must be strictly before each fitting cutoff; validation label_end must be before the outer issue. No later evaluation error chooses that issue\'s pack.')
p('Feature choice minimizes row-count-weighted pooled validation MAE within a model family. Five packs x three windows x four issues = 60 inner fits per family. A separate gate checks >=0.1% median MAE improvement, >=2/3 window wins or ties, and no >2% window MAE loss. Pure-model predictions are retained even when this gate fails.')
p('Each fit keeps features with >=80% finite training coverage and more than one training value. Median imputation, median centering, IQR scaling and clipping to [-8,8] are fitted on all mature training rows. One missing-value indicator is appended per feature, so d base inputs become 2d model inputs.')
p('Target normalization: c = median(y_train), s = max(sample_std(y_train),1), y_scaled = (y-c)/s. Final prediction = max(0, c + s x model_output). No target log transform, sign conditioning or UP/DOWN prediction subtraction is used.')
p('Model estimation uses at most 6,000 evenly spaced mature rows. Preprocessing, target scaling and mean/median baselines use all mature rows. Seed=48; two CPU threads requested; no GPU. GRU uses exact-time snapshots 12/8/4/0 hours earlier from a feature timeline built before target eligibility filtering.')

descriptions={
'hgb':('Histogram gradient boosting','Additive ensemble of regression trees; up to 60 boosting rounds, each tree at most 7 leaves and depth 3. Absolute-error loss.','loss=absolute_error; learning_rate=0.1; max_iter=60; max_leaf_nodes=7; max_depth=3; min_samples_leaf=64; l2_regularization=5; max_bins=255; early_stopping=False; random_state=48.'),
'extra':('ExtraTrees regression','Average of 48 randomized regression trees. Squared-error splitting; maximum depth 10; at least 32 training samples per leaf.','n_estimators=48; criterion=squared_error; max_depth=10; min_samples_leaf=32; min_samples_split=2; max_features=0.8; bootstrap=False; n_jobs=1; random_state=48.'),
'mlp':('Multilayer perceptron','2d inputs -> Dense(16, ReLU) -> Dense(1, identity). Scalar normalized target; squared-error regression.','hidden_layer_sizes=(16,); activation=relu; solver=adam; alpha=1.0; learning_rate_init=0.001; batch_size=256; max_iter=40; early_stopping=False; shuffle=False; random_state=48; beta_1=0.9; beta_2=0.999; epsilon=1e-8; tol=1e-4; n_iter_no_change=10.'),
'gru':('Gated recurrent unit','Four snapshots x 2d inputs -> single-layer GRU(hidden=16) -> last hidden state -> Linear(16,1). Absolute-error loss.','hidden_size=16; num_layers=1; dropout=0; bidirectional=False; 24 epochs; batch_size=256; AdamW lr=0.001; weight_decay=0.03; betas=(0.9,0.999); eps=1e-8; gradient_norm_clip=1; seed=48; no shuffling; CPU float32.')}
for fam in RUNS:
    page(LABEL[fam]+' | architecture and hyperparameters')
    title,architecture,hyper=descriptions[fam]
    p(title,'sub');p(architecture);p(hyper)
    if fam=='hgb':p('Normalized prediction: initial constant plus learning_rate x sum of fitted tree leaf outputs. Saved models report their actual fitted iteration count below.')
    if fam=='extra':p('Normalized prediction: mean of all 48 tree predictions. This is one tree-ensemble family, with no GRU/MLP mixing.')
    if fam=='mlp':p('Normalized prediction: W2 x ReLU(W1 x z + b1) + b2. Each input feature has a scaled value and missing flag. Warnings at 40 epochs mean the iteration budget was reached; validation scores remain the basis for pack selection.')
    if fam=='gru':
        p('GRU gates: r=sigmoid(W_ir x + b_ir + W_hr h_prev + b_hr); z=sigmoid(W_iz x + b_iz + W_hz h_prev + b_hz); n=tanh(W_in x + b_in + r*(W_hn h_prev+b_hn)); h=(1-z)*n+z*h_prev. Output = W_out h_last + b_out. All products inside the gates are elementwise where appropriate.','small')
    rows=[]
    for a in cards[fam]:
        actual=(str(a.get('iterations'))+' rounds' if fam=='hgb' else str(a.get('trees'))+' trees' if fam=='extra' else str(a['epochs'])+' epochs / '+str(a['parameter_count'])+' params' if fam=='mlp' else str(a['parameters'])+' params')
        rows.append([pd.Timestamp(a['issue']).strftime('%b %Y'),a['spec']['pack'],len(a['features']),a['transformed_inputs'],actual,'PASS' if a['accepted'] else 'FAIL'])
    table(['Issue','Feature pack','Base d','Inputs 2d','Fitted architecture','Gate'],rows,[95,135,65,80,285,95])
    p('Training row counts, target centers/scales, exact feature order, all estimator parameters and tensor shapes are in the support model_metadata.json. Gate status is diagnostic; the primary forecast always uses '+LABEL[fam]+'.','small')

page('Quarterly choices and feature-pack search')
rows=[]
for fam in RUNS:
    for a in cards[fam]:
        rows.append([LABEL[fam],pd.Timestamp(a['issue']).strftime('%b'),a['spec']['pack'],f'{a["validation_mae"]:.4f}',a['training_rows'],a['estimator_rows']])
table(['Model','Issue','Chosen pack','Validation MAE','Training rows','Estimator rows'],rows[:8],[90,60,140,145,160,160])
page('Quarterly choices | neural models')
table(['Model','Issue','Chosen pack','Validation MAE','Training rows','Estimator rows'],rows[8:],[90,60,140,145,160,160])
page('Candidate feature packs | what was compared')
table(['Pack','Count','Contents'],[
['market',13,'11 inherited F4Z market variables and 2 historical price-volatility variables.'],
['prior_price',235,'Legacy name: lagged/rolling transformations of market AND price inputs, plus calendar and longer price features. It is not exclusively price data.'],
['new_price',48,'New causal 5-minute-return features defined in the following pages.'],
['all',296,'Union of market13 + prior_price235 + new_price48.'],
['market_new',61,'market13 + new_price48.']],[135,65,555])
p('The winning forecasts used 61 distinct base features in total: the market13 and new_price48 sets. The other candidate packs were tested but never selected for the four primary family forecasts. Exact ordered membership of all 296 candidates is supplied in candidate_feature_packs.json.')
p('These experiments select feature groups, not the individual feature with the largest marginal error reduction. Correlated features, redundant transforms and interaction effects remain. For example RMS_h = RV_h/sqrt(12h), so these two columns are scaled copies at a fixed horizon.')

market=contracts['hgb']['packs']['market']
meaning={
'PRICE__vol_ratio_4h_24h':'Inherited ratio of historical 4h and 24h volatility; upstream definition retained.',
'PRICE__realized_vol_168h':'Inherited historical 168h realized volatility; upstream definition retained.'}
for i in range(0,len(market),7):
    page('Exact market features | '+str(i+1)+'-'+str(min(i+7,len(market))))
    rows=[]
    for feat in market[i:i+7]:
        if feat in meaning:description=meaning[feat]
        elif 'historical_active_rate' in feat:description='Historical peer liquidation activity frequency; not current liquidation size.'
        elif 'flow_activity' in feat:description='Aggressive flow activity percentile; not signed buy-minus-sell flow.'
        elif 'basis_level_rz' in feat:description='Historical basis-level robust standardized value.'
        elif 'funding_persistence' in feat:description='Funding persistence percentile.'
        elif 'venue_breadth' in feat:description='Liquidation venue breadth percentile.'
        elif 'venue_fragmentation' in feat:description='Liquidation venue fragmentation percentile.'
        else:description='Liquidation temporal concentration percentile.'
        rows.append([market.index(feat)+1,feat,description])
    table(['ID','Exact base-feature identifier','Meaning'],rows,[35,440,280],True)
p('F4Z percentile/reference distributions and upstream normalization are inherited from the supplied data. Their generating formulas were not independently reconstructed in this report. Exact identifiers are preserved.','small')

page('New features | formula notation and safeguards')
p('Let r_i = 10,000 x ln(P_i/P_(i-5min)) be a canonical 5-minute log-bps return, indexed by its ending time. For decision time t, window W_h contains endpoints t-h <= i < t. Windows exclude i=t and every future return.')
p('For h in {4,24,48,168,336} hours, require all 12h finite returns. Let N=sum r_i, A=sum abs(r_i), Q=sum r_i^2, U=sum max(r_i,0)^2, J=max abs(r_i). Missing/infinite feature values are left missing until training-fitted preprocessing.')
table(['Name family','Formula','Meaning'],[
['rv','sqrt(Q)','Realized volatility'],['net','N','Signed cumulative log return'],['abs_net','abs(N)','Absolute cumulative log return'],['abs_sum','A','Total absolute movement'],['efficiency','abs(N)/max(A,1e-12)','Trend efficiency'],['up_variance_share','U/max(Q,1e-12)','Upside share of squared movement'],['jump_share','J^2/max(Q,1e-12)','Largest-return concentration'],['rms','sqrt(Q)/sqrt(12h)','Root mean squared 5-minute return'],['positive_share','mean(1[r_i>0])','Fraction of positive returns']],[150,235,370])
p('Five lookbacks x nine families = 45 features. Three additional ratios use (RV_h/sqrt(h))/max(RV_168/sqrt(168),1e-9), for h=4,24,48. Total new features: 48.','small')
for start in range(0,len(formulas),12):
    page(f'Exact new-feature formulas | {start+1}-{min(start+12,len(formulas))}')
    table(['Exact feature name','Implemented formula'],[[r.feature,r.formula] for r in formulas.iloc[start:start+12].itertuples()],[290,465],True)

page('Exact feature membership | market inputs')
p('Jan/Apr/Jul/Oct denote the issue where that feature is used; the October model also covers final January 2025 observations. GRU takes each listed feature plus its missing flag at all four snapshots.')
rows=[[r['feature'],r['hgb'],r['extra'],r['mlp'],r['gru']] for r in membership if r['feature'] in market]
for start in range(0,len(rows),7):
    if start:page('Exact feature membership | market inputs continued')
    table(['Base feature','HGB','ExtraTrees','MLP','GRU'],rows[start:start+7],[400,90,90,90,85],True)
newrows=[[r['feature'],r['hgb'],r['extra'],r['mlp'],r['gru']] for r in membership if r['feature'].startswith('NET48__')]
for start in range(0,len(newrows),16):
    page(f'Exact feature membership | new inputs {start+1}-{min(start+16,len(newrows))}')
    table(['Base feature','HGB','ExtraTrees','MLP','GRU'],newrows[start:start+16],[315,110,110,110,110],True)

page('All forecast variants | historical reference')
p('The primary ranking contains only the four independent families and two baselines. For completeness, the earlier mixed-family run and gate-based audit policies are listed here separately. They are not additional trained model architectures.')
variantrows=[]
for fam,f in data.items():
    m=metric(f,'guarded_selected');variantrows.append([LABEL[fam]+' gated audit',f'{m["mae"]:.6f}',f'{m["rmse"]:.6f}','May substitute median; excluded from pure-model ranking.'])
old=EXP/'net_magnitude_48h_v1/20261116T044116476061Z'
done=js(old/'COMPLETE.json');assert sha(track(old/'predictions.csv'))==done['prediction_sha256']
f=csv(old/'predictions.csv');f.decision_time=pd.to_datetime(f.decision_time,utc=True)
pd.testing.assert_frame_equal(reference[['decision_time','actual']],f.sort_values('decision_time')[['decision_time','actual']].reset_index(drop=True))
for col in ['candidate','selected']:
    m=metric(f,col);variantrows.append(['Earlier mixed '+col,f'{m["mae"]:.6f}',f'{m["rmse"]:.6f}','Historical reference; model family changes across issues.'])
table(['Forecast policy','MAE','RMSE','Status'],variantrows,[195,110,110,340])

page('Evidence, reproducibility and limits')
p('Audit performed: five prediction hashes verified (four pure runs plus original mixed run); all 16 saved pure-family model hashes verified; 18,385 unique timestamps and identical actuals/baselines aligned; pooled and period errors recomputed directly from prediction CSVs. Saved estimator parameters, input shapes and GRU state tensors inspected. All 240 pure-family validation trials included in support CSVs. No training or new inference was performed to create this report.')
p('Saved COMPLETE files report successful model replay; replay was not rerun for this report. The GRU saved report and CSV recomputation differ by about 4.7e-8 MAE, consistent at displayed precision. Run snapshots and artifacts, rather than later edited training code, are used as evidence.')
p('Each family used its validation-best feature pack even where the acceptance gate failed. Only the October HGB and October GRU entries pass the saved gate. GRU is the lowest-MAE observed pure-family forecast, not independently confirmed superior. The historical 2022-2025 evaluation has already informed research choices; no statistical significance or untouched-test claim is made.')
p('This report establishes feature-pack comparisons and exact fitted inputs. It does not establish individual causal importance or the globally optimal feature subset. MLP convergence warnings indicate that some fits reached the iteration budget; increasing epochs would be a new experiment.')
for fam,stamp in RUNS.items():p(f'{LABEL[fam]} source: Code/experiments/net_magnitude_48h_{fam}_only_v1/{stamp}/','small')
p('Support: rankings and period metrics; exact feature membership; new-feature formulas; candidate pack inventory; full model metadata; four validation ledgers; four contracts; audit hashes. Builder: Code/build_net_magnitude_48h_report.py.','small')

def footer(canvas,doc):
    canvas.setStrokeColor(TEAL);canvas.line(40,34,802,34)
    canvas.setFont('Helvetica',9);canvas.setFillColor(NAVY)
    canvas.drawString(40,21,'BTCUSDT | 48h net magnitude | 16 Nov 2026 | log bps')
    canvas.drawRightString(802,21,str(doc.page))

doc=SimpleDocTemplate(str(PDF),pagesize=landscape(A4),leftMargin=43,rightMargin=43,topMargin=32,bottomMargin=48,title='BTC 48h Net Magnitude - Models, Features and Rankings',author='ScaleWave research')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(str(PDF));text='\n'.join(page.extract_text() for page in reader.pages)
assert all(v in text for v in ['181.870','182.909','ExtraTrees','NET48__rv_ratio_48_168','GRU'])
assert all(feat in text.replace('\n','') for feat in used if feat.startswith('NET48__'))
dump(SUP/'audit.json',dict(pdf=str(PDF),pages=len(reader.pages),prediction_runs_verified=5,model_hashes_verified=16,observations=18385,distinct_used_base_features=len(used),new_features=48,hashes=hashes,pdf_sha256=sha(PDF),training_performed=False,inference_performed=False))
print(json.dumps(dict(pdf=str(PDF),pages=len(reader.pages),support=str(SUP))))
