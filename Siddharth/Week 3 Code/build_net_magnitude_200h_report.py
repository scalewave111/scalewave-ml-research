"""Audit saved 200h results and create report. No training or predictions."""
from pathlib import Path
import json,hashlib,shutil,sys
from xml.sax.saxutils import escape
import joblib,numpy as np,pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib import colors
from pypdf import PdfReader
from fit_net_magnitude_200h_refined import NeuralHead

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'Report';EXP=ROOT/'Code/experiments'
SUP=OUT/'net_magnitude_200h_report_support_2026-11-16';SUP.mkdir(exist_ok=True)
PDF=OUT/'BTC_200h_Net_Magnitude_Models_Features_Rankings_2026-11-16.pdf'
RUNS={'hgb':'20261116T113845228744Z','extra':'20261116T113920764170Z','mlp':'20261116T114232727007Z','gru':'20261116T114411205180Z'}
LABEL={'hgb':'HGB','extra':'ExtraTrees','mlp':'MLP','gru':'GRU'}
hashes={};data={};contracts={};plans={};cards={};rows=[];members=[];trials_all=[];reference=None
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def track(p):hashes[str(p)]=sha(p);return p
def js(p):return json.loads(track(p).read_text(encoding='utf-8-sig'))
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,default=str,allow_nan=False),encoding='utf-8')
def metric(f,col):
    e=f[col].to_numpy()-f.actual.to_numpy()
    return dict(rows=len(e),mae=float(abs(e).mean()),rmse=float(np.sqrt((e*e).mean())),bias=float(e.mean()))
for fam,stamp in RUNS.items():
    path=EXP/f'net_magnitude_200h_{fam}_refined_fast_v1'/stamp
    done=js(path/'COMPLETE.json');assert done['status']=='COMPLETE' and sha(track(path/'predictions.csv'))==done['prediction_sha256']
    f=pd.read_csv(path/'predictions.csv');f.decision_time=pd.to_datetime(f.decision_time,utc=True);f.issue=pd.to_datetime(f.issue,utc=True)
    assert len(f)==17473 and f.decision_time.is_unique and np.isfinite(f[['actual','candidate','selected','mean','median']]).all().all()
    if reference is None:reference=f[['decision_time','actual','mean','median']]
    else:pd.testing.assert_frame_equal(reference,f[['decision_time','actual','mean','median']])
    data[fam]=f;contracts[fam]=js(path/'CONTRACT.json');plans[fam]=js(path/'selections.json');cards[fam]=[]
    saved=pd.read_csv(track(path/'metrics.csv'));trials=js(path/'validation_trials.json');assert len(trials)==1600
    for period,g in [('ALL',f)]+[(str(k),v) for k,v in f.groupby('issue')]:
        for col in ['candidate','selected','mean','median']:
            m=metric(g,col);r=saved[(saved.period==period)&(saved.model==col)].iloc[0]
            np.testing.assert_allclose([m['mae'],m['rmse'],m['bias']],[r.mae,r.rmse,r.bias],rtol=0,atol=1e-5)
            rows.append(dict(family=fam,period=period,forecast=col,**m))
    for plan in plans[fam]:
        group=[t for t in trials if t['issue']==plan['issue']]
        best=min(group,key=lambda t:(t['mae'],t['score'],t['id'],t['anchor'],t['weight'],t['head_weight']))
        assert best==plan['candidate'] and plan['fallback'] and plan['selected'] is None
        assert not any(t['accepted'] for t in group)
        for record in plan['models']:
            mp=track(path/record['artifact']);assert sha(mp)==record['sha256']
            bundle=joblib.load(mp);assert bundle['prep']['cols']==record['features']
            assert pd.Timestamp(bundle['max_label_end'])<pd.Timestamp(plan['issue'])
            heads=[]
            for head in bundle['heads']:
                if hasattr(head,'get_params'):
                    params=head.get_params(deep=False);extra={}
                    if fam=='hgb':extra={'fitted_iterations':head.n_iter_}
                    if fam=='extra':extra={'fitted_trees':len(head.estimators_)}
                    heads.append(dict(parameters=params,**extra))
                else:heads.append(dict(kind=head.kind,config=head.config,objective=head.objective,inputs=head.inputs,state_shapes={k:list(v.shape) for k,v in head.state.items()},parameter_count=sum(v.numel() for v in head.state.values())))
            card=dict(issue=plan['issue'],candidate=plan['candidate'],artifact=record['artifact'],sha256=record['sha256'],features=record['features'],training_rows=bundle['training_rows'],estimator_rows=bundle['estimator_rows'],max_label_end=bundle['max_label_end'],center=bundle['center'],scale=bundle['scale'],inputs=2*len(record['features']),heads=heads)
            cards[fam].append(card)
            for c in record['features']:members.append(dict(family=fam,issue=plan['issue'],feature=c))
    for name in ['CONTRACT.json','selections.json','validation_trials.json','feature_dictionary.csv']:
        shutil.copyfile(track(path/name),SUP/f'{fam}_{name}')
    trials_all += [dict(family=fam,**t) for t in trials]
pd.DataFrame(rows).to_csv(SUP/'period_metrics.csv',index=False)
pd.DataFrame(members).to_csv(SUP/'exact_feature_membership.csv',index=False)
dump(SUP/'model_metadata.json',cards);dump(SUP/'all_validation_trials.json',trials_all)
dictionary=pd.read_csv(SUP/'hgb_feature_dictionary.csv').fillna('');assert len(dictionary)==80 and dictionary.feature.is_unique
ranking=[]
for fam in RUNS:ranking.append(dict(model=LABEL[fam]+' candidate',**metric(data[fam],'candidate')))
for col in ['mean','median']:ranking.append(dict(model=col.title()+' baseline',**metric(data['hgb'],col)))
ranking=pd.DataFrame(ranking);ranking['MAE rank']=ranking.mae.rank(method='min').astype(int);ranking['RMSE rank']=ranking.rmse.rank(method='min').astype(int)
ranking=ranking.sort_values('mae');ranking.to_csv(SUP/'model_rankings.csv',index=False)
selection=dict(selected_by='User instruction on 2026-11-16',family='hgb',forecast_column='candidate',run=RUNS['hgb'],automated_guard_passed=False,refit=False,deployed=False,policy='Quarter-specific HGB L1/L2 head blend plus training baseline; exact weights in quarterly metadata',quarters=cards['hgb'])
dump(SUP/'selected_hgb.json',selection)

# Reuse the established 48h report typography and table design without executing its builder.
layout=(ROOT/'Code/build_net_magnitude_48h_report.py').read_text(encoding='utf-8');a=layout.index('NAVY=');z=layout.index("page('48-hour",a)
exec(layout[a:z],globals())

page('200-hour net magnitude | selected HGB report')
p('BTCUSDT - models, features and rankings','title')
p('16 November 2026 | 17,473 aligned evaluation origins | All errors in log bps')
p('User selection: refined HGB candidate from run '+RUNS['hgb']+'. This records the saved quarterly policy, with no retraining or deployment.')
p('HGB candidate MAE 374.606082: 0.67% below mean and 1.88% below median. RMSE 531.086120 exceeds mean 499.616543 by 6.30%. No simultaneous mean-MAE/RMSE win.')
table(['Forecast','MAE','RMSE','MAE rank','RMSE rank'],[[r['model'],f"{r['mae']:.6f}",f"{r['rmse']:.6f}",r['MAE rank'],r['RMSE rank']] for r in ranking.to_dict('records')],[245,150,150,100,110])
p('All four automated selected policies fell back to mean in all quarters. User-selected HGB refers to the diagnostic candidate column, not the automated selected column. Candidate selection uses earlier validation; choosing its family now follows exposed historical evaluation.','small')

page('Target, population and evidence limits')
p('Y = abs(10000 * ln(P[t+200h]/P[t])). It is endpoint magnitude, not maximum UP excursion, DOWN excursion, or their difference. Lower MAE/RMSE is better; bias is prediction minus actual.')
p('Canonical BTCUSDT horizon_minutes=12000 labels were checksum-verified by the original run. Original grid 48,096 rows; 47,088 eligible, with 608 missing/invalid labels and 400 source-binding crossings excluded. Evaluation contains 17,473 aligned origins.')
p(f"Observed evaluation spans {reference.decision_time.min()} through {reference.decision_time.max()}. Jan/Apr/Jul/Oct 2024 expanding fits; October continues into January 2025.")
p('The original source availability masks remain inherited. Required market sources are read directly from original masked X_train/X_development, not the reduced legacy feature inventory. Canonical/raw source files were not changed.')
p('Saved prediction/model hashes and recomputed scores are verified in this report. Original training reports serialization replay success; replay and inference were not rerun here. Historical data were repeatedly exposed during feature/model refinement; these are not independent confirmation scores.')

page('Selected quarterly HGB policy')
table(['Issue','Pack / inputs','L2 weight q','Model weight w','Anchor','Guard'],[[c['issue'][:10],c['candidate']['spec']['pack']+' / '+str(len(c['features'])),c['candidate']['head_weight'],c['candidate']['weight'],c['candidate']['anchor'],'Failed'] for c in cards['hgb']],[100,220,100,115,110,110])
p('For the user-selected HGB candidate, q=0 in every issue: only the absolute-error head contributes. The squared-error heads are saved comparison components with zero candidate weight.')
p('Head blend H=(1-q)*max(0,L1*scale+center)+q*max(0,L2*scale+center). Candidate prediction=w*H+(1-w)*training anchor. This is one HGB family with two loss heads and baseline shrinkage; it is not a single unblended tree estimator.')
p('At inference, fewer than 80% finite retained base inputs makes each head return the training median before the baseline blend. For GRU comparisons this coverage condition applies to every snapshot.')
table(['Issue','Mature rows','Estimator rows','Latest training label end','Center / scale'],[[c['issue'][:10],c['training_rows'],c['estimator_rows'],str(c['max_label_end']),f"{c['center']:.4f} / {c['scale']:.4f}"] for c in cards['hgb']],[90,100,100,285,180],True)

for i in [0,2]:
    page('Quarterly scores | '+('January and April' if i==0 else 'July and October'))
    table(['Issue','Forecast','Rows','MAE','RMSE'],[[pd.Timestamp(issue).strftime('%b %Y'),name,metric(g,col)['rows'],f"{metric(g,col)['mae']:.4f}",f"{metric(g,col)['rmse']:.4f}"] for issue in list(data['hgb'].issue.unique())[i:i+2] for name,g,col in [(LABEL[f]+' candidate',data[f][data[f].issue==issue],'candidate') for f in RUNS]+[(b.title(),data['hgb'][data['hgb'].issue==issue],b) for b in ['mean','median']]],[115,220,100,160,160],True)

page('Validation, acceptance and refinement history')
p('Four earlier three-month windows start 15/12/9/6 months before each issue. Training label_end must be strictly before fit cutoff. Validation label_end must be strictly before its window end and before the outer issue. Full-timeline features precede outcome filtering.')
p('Each family evaluates four packs with paired L1/L2 heads. Model blend weights .05/.1/.2/.25/.35/.5/.65/.75/.9/1, head weights 0/.25/.5/.75/1, mean/median anchors: 100 combinations per fitted pack; 1,600 trial records per family. Fits are reused across blends.')
p('Refined candidate and accepted settings minimize pooled earlier MAE, then worst relative MAE/RMSE score and deterministic identifiers. Acceptance requires >=0.1% pooled gain against BOTH mean/median MAE and mean RMSE, >=3/4 window wins/ties and no >2% window loss for each comparison. No trial passed in any issue/family. Mean fallback remains the original automated policy.')
p('Earlier UP-style HGB candidate MAE 378.059030; endpoint-feature HGB 376.890379; refined HGB 374.606082. Feature additions, blend grids and validation ranking changed together; the improvement cannot be assigned causally to an individual new feature.')

page('Architectures and fast training settings')
table(['Family','Architecture / fitted settings'],[
['HGB','Two heads: absolute_error and squared_error; 40 iterations, depth3,7leaves,minleaf128,rate.05,L2=10,bins63,no early stopping.'],
['ExtraTrees','Two heads: absolute_error/squared_error;24trees,depth4,minleaf32,maxfeatures.75,bootstrap up to1024rows/tree,n_jobs1.'],
['MLP','Two independent32hidden ReLU heads,16epochs,L1/MSE,AdamWlr.001,decay.1,batch256,clipgrad1.'],
['GRU','Two independent16hidden GRU heads plus linear output;12epochs,L1/MSE,AdamWlr.001,decay.03,batch256,clipgrad1; snapshots60/40/20/0h.']],[125,630])
p('Seed48,CPUthreads2; up to2,000 evenly spaced mature rows per estimator. All mature rows supply feature-quality screening, preprocessing and target/baseline statistics. Default signed-log input transform. No random early-stopping split.')
p('For a retained feature: median imputation, z=clip((x-training median)/max(training IQR,1e-6),-8,8); signedlog(z)=sign(z)*ln(1+abs(z)); append original missing flag. Retain >=80% finite training coverage, >=1,000 finite values and variation. Target=(Y-training median)/max(training std,1).')

page('Saved HGB head parameters and equations')
for c in cards['hgb']:
    p(c['issue'][:10]+' | '+str(c['inputs'])+' transformed inputs | '+c['artifact'],'sub')
    p('L1 / L2 iterations: '+str([h['fitted_iterations'] for h in c['heads']])+'. Target center '+f"{c['center']:.6f}"+', scale '+f"{c['scale']:.6f}"+'.','small')
p('Each head is an additive boosted tree model f(z)=intercept+sum(tree_m(z)). Histogram splits approximate the specified loss with the saved regularization and leaf constraints. Exact fitted estimator parameters are in model_metadata.json; original bundles remain at their experiment paths.')

page('Candidate packs and formula conventions')
table(['Pack','Base inputs','Meaning'],[[name,len(cols),{'volatility_endpoint':'Volatility scale, endpoint proxies and past magnitude persistence','trend_range':'Momentum, range, efficiency, variance share and endpoint features','slow_market':'Volatility/endpoint features plus funding,basis,OI,flow,liquidation state','endpoint_refined':'Trend-range control plus12endpoint quantile/persistence/regime inputs'}[name]] for name,cols in contracts['hgb']['packs'].items()],[185,95,475])
p('80 distinct candidate inputs; exact used feature names and quarter/family memberships follow. Generated dictionary entries are not evidence every feature was used. Historical quantile windows overlap; overlapping blocks are not independent samples.')
p('r denotes canonical endpoint-indexed5min log-bps returns. Completed-hour returns aggregate endpoints strictly before the hour timestamp. LONG uses the latest completed hour within59m59s; raw END features use [t-h,t). Rolling market STATE means exclude origin, require80% expected half-hour snapshots. No predictor uses future outcomes.')
p('F4Z source values/normalization are inherited, not reconstructed independently. Liquidation activity features measure historical activity frequency; their names do not mean current liquidation size or signed trade flow.')

used=sorted(set(m['feature'] for m in members));selected_used=sorted(set(c for card in cards['hgb'] for c in card['features']))
for start in range(0,len(used),12):
    page('Exact fitted inputs and quarterly membership | '+str(start//12+1))
    table(['Feature','HGB','ExtraTrees','MLP','GRU'],[[feat]+[','.join(pd.Timestamp(c['issue']).strftime('%b') for c in cards[f] if feat in c['features']) or '-' for f in RUNS] for feat in used[start:start+12]],[415,85,85,85,85],True)
for start in range(0,len(dictionary),5):
    page('Feature formulas and source definitions | '+str(start//5+1))
    for row in dictionary.iloc[start:start+5].to_dict('records'):
        p(row['feature'],'sub');p(row.get('formula',''),'small')
        if row.get('sources'):p('Source: '+row['sources'],'small')

page('Provenance, verification and retained artifacts')
p('Verified COMPLETE for4runs,4prediction hashes and16paired-model bundle hashes. Recomputed all pooled and quarterly MAE/RMSE/bias; aligned17,473unique timestamps/actuals/baselines. Audited6,400validation records and candidate minima; all16quarter/family guards failed. Loaded saved model metadata without generating predictions.')
p(f'Selected HGB uses {len(selected_used)} distinct base inputs across four issues; all-family fitted union {len(used)}. These counts describe fitted membership, not feature importance or independent predictive contribution.')
for fam,stamp in RUNS.items():p(LABEL[fam]+': Code/experiments/net_magnitude_200h_'+fam+'_refined_fast_v1/'+stamp,'small')
p('Support includes user selection,exact feature memberships,all80formulas,all validation ledgers,contracts,quarterly scores,model metadata and source/audit hashes. Raw datasets and historical results are unchanged. No training,inference,deployment or automatic promotion performed for this report.')

def footer(canvas,doc):
    canvas.setStrokeColor(TEAL);canvas.line(42,34,800,34)
    canvas.setFont('Helvetica',9);canvas.setFillColor(NAVY);canvas.drawString(42,21,'BTCUSDT | 200h net magnitude | Historical exposed evaluation')
    canvas.drawRightString(800,21,str(doc.page))
SimpleDocTemplate(str(PDF),pagesize=landscape(A4),rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=48).build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(PDF);assert all(page.extract_text().strip() for page in reader.pages)
dump(SUP/'audit.json',dict(hashes=hashes,pages=len(reader.pages),rows=17473,models_verified=16,validation_records=6400,selected_hgb_feature_union=len(selected_used),all_family_feature_union=len(used),training=False,inference=False,replay='Reported by original runs; not rerun'))
shutil.copyfile(__file__,SUP/Path(__file__).name)
print(json.dumps(dict(pdf=str(PDF),pages=len(reader.pages),support=str(SUP),selected_hgb_features=len(selected_used))))
