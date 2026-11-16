"""Audit all completed200h experiments and create a PDF; no training/inference."""
from pathlib import Path
import json,hashlib,shutil,zipfile,re,joblib
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import landscape,A4
from reportlab.lib import colors
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]; EXP=ROOT/'Code/experiments'
OUT=ROOT/'Report';SUP=OUT/'200h_report_support_2026-11-14'
PDF=OUT/'BTC_200h_Models_Features_Rankings_2026-11-14.pdf'
RUNS={
 'A':('Initial feature search','excursion_pipeline_200h_v1/20261113T115534387046Z/search/20261113T115543583910Z'),
 'B':('Price-block follow-up','excursion_pipeline_200h_v1/20261113T115534387046Z/price_followup/20261113T121536996657Z'),
 'C':('Fresh long-feature search','long_features_200h_v1/20261113T123706785745Z'),
 'D':('Market HGB/MLP/GRU','magnitude_models_200h_v1/20261114T131905974842Z'),
 'E':('Joint MAE/RMSE refinement','dual_objective_200h_v1/20261114T135118300960Z'),
 'F':('Transformed joint refinement','transformed_dual_200h_v1/20261114T141213076180Z')}
PRIMARY={'A':'prediction','B':'selected','C':'selected','D':'selected','E':'selected','F':'selected'}
SUP.mkdir(parents=True,exist_ok=True)
hashes={}
def track(p):hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def js(p):return json.loads(track(p).read_text(encoding='utf-8-sig'))
def csv(p):return pd.read_csv(track(p))
def metrics(y,p):
 return dict(MAE=float(np.abs(y-p).mean()),RMSE=float(np.sqrt(np.square(y-p).mean())),IC=float(pd.Series(y).corr(pd.Series(p),method='spearman')) if np.ptp(y)>0 and np.ptp(p)>0 else float('nan'),Bias=float(np.mean(p-y)))
def side(t):return t.split('_')[0].upper()
data={};cards={};contracts={};audit=[];allmetrics=[];policies=[];vectors={};periodrows=[]
for label,(name,relative) in RUNS.items():
 path=EXP/relative;done=js(path/'COMPLETE.json')
 assert done['status']=='COMPLETE' and hashlib.sha256((path/'predictions.csv').read_bytes()).hexdigest()==done['predictions_sha256']
 f=csv(path/'predictions.csv').sort_values(['target','decision_time']).reset_index(drop=True)
 assert not f.duplicated(['target','decision_time']).any() and f.groupby('target').size().eq(17873).all()
 if data:
  assert f[['target','decision_time','issue']].equals(data['A'][['target','decision_time','issue']])
  np.testing.assert_allclose(f.actual,data['A'].actual,atol=1e-10,rtol=0)
  for c in ['training_median','training_mean']:np.testing.assert_allclose(f[c],data['A'][c],atol=1e-8,rtol=0)
 data[label]=f;cards[label]=js(path/'selections.json');contracts[label]=js(path/'CONTRACT.json')
 for filename in ['COMPLETE.json','CONTRACT.json','selections.json']:shutil.copyfile(path/filename,SUP/f'{label}_{filename}')
 for p in path.glob('*.py'):track(p)
 source_metrics=csv(path/('metrics_all_variants.csv' if label=='B' else 'metrics.csv'))
 source_metrics.to_csv(SUP/f'{label}_saved_metrics.csv',index=False)
 for target,g in f.groupby('target'):
  for col in f.columns:
   if col in ['decision_time','issue','target','actual','period'] or not pd.api.types.is_numeric_dtype(f[col]):continue
   for period,part in [('ALL',g)]+list(g.groupby('period')):
    usable=np.isfinite(part[col]);sub=part.loc[usable]
    if not len(sub):continue
    values=metrics(sub.actual.to_numpy(),sub[col].to_numpy())
    allmetrics.append(dict(run=label,target=side(target),forecast=col,period=period,rows=len(sub),full_coverage=len(sub)==len(part),**values))
    saved=source_metrics[(source_metrics.target==target)&(source_metrics.model==col)&(source_metrics.period==period)]
    if len(saved)==1:
     for lower,upper in [('mae','MAE'),('rmse','RMSE'),('ic','IC')]:np.testing.assert_allclose(saved.iloc[0][lower],values[upper],rtol=1e-8,atol=1e-8,equal_nan=True)
   if col==PRIMARY[label] or (col=='candidate' and label in 'CDEF'):
    v=g[col].to_numpy();assert np.isfinite(v).all()
    row=dict(run=label,target=side(target),forecast=col,name=name,role='candidate' if col=='candidate' else 'selected',**metrics(g.actual.to_numpy(),v))
    policies.append(row);vectors[(side(target),label,col)]=v
  for period,part in g.groupby('period'):
   periodrows.append(dict(run=label,target=side(target),period=period,**metrics(part.actual.to_numpy(),part[PRIMARY[label]].to_numpy())))
 audit.append(dict(run=label,name=name,path=relative,predictions_sha256=done['predictions_sha256'],rows=len(f),hash_verified=True,replay_reported=done.get('replay_passed',done.get('original_replay_passed'))))
for target,g in data['A'].groupby('target'):
 for col in ['training_median','training_mean']:
  policies.append(dict(run='BASE',target=side(target),forecast=col,name=col.replace('_',' ').title(),role='baseline',**metrics(g.actual.to_numpy(),g[col].to_numpy())))
  vectors[(side(target),'BASE',col)]=g[col].to_numpy()
pol=pd.DataFrame(policies)
for target in ['UP','DOWN']:
 ix=pol.target.eq(target)
 for metric,ascending in [('MAE',True),('RMSE',True),('IC',False)]:pol.loc[ix,metric+'_rank']=pol.loc[ix,metric].rank(method='min',ascending=ascending)
pol=pol.sort_values(['target','MAE','run']);pol.to_csv(SUP/'policy_rankings.csv',index=False)
allm=pd.DataFrame(allmetrics);allm.to_csv(SUP/'all_forecasts_period_metrics.csv',index=False)
pd.DataFrame(periodrows).to_csv(SUP/'selected_period_metrics.csv',index=False)
# Full-period fixed variants join the policy ranking, with duplicate predictions collapsed.
full=[];partial=[]
for target in ['UP','DOWN']:
 seen=[]
 candidates=[(r['run'],r['forecast'],r['role']) for r in policies if r['target']==target]
 candidates += [('B',c,'fixed diagnostic') for c in data['B'].columns if c not in ['decision_time','issue','target','actual','period','selected','training_mean','training_median','original']]
 for run,col,role in candidates:
  g=data['A' if run=='BASE' else run];g=g[g.target.str.startswith(target.lower())]
  values=g[col].to_numpy(float);valid=np.isfinite(values)
  if not valid.all():
   partial.append(dict(target=target,run=run,forecast=col,rows=int(valid.sum())))
   continue
  match=next((r for v,r in seen if np.allclose(v,values,atol=1e-7,rtol=0)),None)
  if match is not None:match['aliases']+='; '+run+'.'+col;continue
  row=dict(target=target,run=run,forecast=col,role=role,aliases=run+'.'+col,**metrics(g.actual.to_numpy(),values))
  seen.append((values,row));full.append(row)
full=pd.DataFrame(full).sort_values(['target','MAE','run'])
for target in ['UP','DOWN']:
 ix=full.target.eq(target)
 for m,a in [('MAE',True),('RMSE',True),('IC',False)]:full.loc[ix,m+'_rank']=full.loc[ix,m].rank(method='min',ascending=a)
full.to_csv(SUP/'all_full_coverage_rankings.csv',index=False);pd.DataFrame(partial).to_csv(SUP/'partial_variant_coverage.csv',index=False)

# Read saved estimators for metadata only. No fitting or prediction occurs.
packs={};features={};nodes=[];roots=[];model_hashes=[]
def pack(cols):
    key=tuple(cols)
    if key not in packs:packs[key]=f'P{len(packs)+1:03d}'
    features[packs[key]]=list(cols)
    return packs[key]
def node(kind,cols,formula,settings,children=()):
    ident=f'N{len(nodes)+1:03d}'
    nodes.append(dict(id=ident,kind=kind,pack=pack(cols),features=cols,formula=formula,settings=settings,children=list(children)))
    return ident

def leaf(m,cols):
    if 'constant' in m:return node('Constant',[],str(m['constant']),'Training-only constant')
    if 'state' in m:
        cfg=m['config'];d=len(cols)//4 if cfg['kind']=='gru' else len(cols)
        return node(('MLP32/16' if cfg['kind']=='mlp' and cfg['hidden']==32 else cfg['kind'].upper()+str(cfg['hidden'])),cols,
            'm + w*0.75*max(m,1)*tanh(Linear(last hidden)); coverage<80% => m',
            dict(**cfg,epochs=24,batch=512,cap=4096,seed=48,gradient_clip=1,raw_inputs_per_step=d,median=m['median']))
    e=m['model']
    return node(type(e).__name__,cols,'max(0, fitted estimator(preprocess(x)))',e.get_params())

def market_leaf(m,spec):
    a=leaf(m,m['features'])
    if 'constant' not in m:
        nodes[-1]['settings']['weight']=spec.get('weight',1.)
    return a

def anchor(a):
    if a.get('kind')=='dual':return dual(a['model'],'candidate')
    if a.get('kind')=='neural':return market_leaf(a['model'],a['spec'])
    return leaf(a['model'],a['features'])

def dual(bundle,branch):
    a=anchor(bundle['anchor']);spec=bundle['plan'][branch]
    if spec['model']=='anchor':return a
    m=bundle[branch];ls=[]
    for name in ['median','mean']:
        e=m[name]
        ls.append(node(type(e).__name__+(' L1' if name=='median' else ' L2'),m['features'],
            'max(0, unit*estimator(transformed x)); missing coverage<80% => '+a,
            dict(params=e.get_params(),transform=m.get('config',{}).get('transform','standard'),unit=m.get('unit',1))))
    u=spec['update_weight'];w=spec['mean_weight']
    cols=list(dict.fromkeys(nodes[int(a[1:])-1]['features']+m['features']))
    return node('Blend',cols,f'{1-u:g}*{a} + {u*(1-w):g}*{ls[0]} + {u*w:g}*{ls[1]}',spec,[a]+ls)

def load_model(path,expected):
    digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==expected
    model_hashes.append(dict(path=str(path.relative_to(ROOT)),sha256=digest));track(path)
    return joblib.load(path)
for label,(_,relative) in RUNS.items():
    path=EXP/relative
    for s in cards[label]:
        target,issue=s['target'],s['issue'];month=issue[:7]
        folder=path/(target+'/'+month if label in 'DEF' else month+'_'+target)
        if label=='B':
            card=js(folder/'selection.json');v=card['variants'][card['selected_variant']]
            saved=load_model(folder/v['artifact'],v['artifact_sha256'])
        else:saved=load_model(folder/'model.joblib',js(folder/'MODEL_HASH.json')['sha256'])
        for branch in (['selected','candidate'] if label in 'CDEF' else ['selected']):
            start=len(nodes)
            if label in 'AB':root=leaf(saved['model'],saved['features'])
            elif label=='C':
                root=leaf(saved['model'],saved['features']) if branch=='candidate' or saved['policy']['accepted'] else node('Constant',[],str(saved['constant']),saved['policy']['fallback'])
            elif label=='D':root=market_leaf(saved[branch],saved['plan'][branch])
            else:root=dual(saved,branch)
            roots.append(dict(run=label,target=side(target),issue=month,forecast=branch,root=root,nodes=[n['id'] for n in nodes[start:]],pack=nodes[int(root[1:])-1]['pack']))

base_names=sorted(set(re.sub(r'^SEQ[0-9]+__','',c) for cols in features.values() for c in cols))
fid={c:f'F{i+1:03d}' for i,c in enumerate(base_names)}
# Export exact components and estimator parameters, ordered inputs and transforms.
(SUP/'model_components.json').write_text(json.dumps(nodes,indent=2,default=str))
pd.DataFrame(roots).to_csv(SUP/'quarterly_model_roots.csv',index=False)
pd.DataFrame([dict(pack=k,ordered_columns=json.dumps(v)) for k,v in features.items()]).to_csv(SUP/'feature_packs.csv',index=False)
pd.DataFrame([dict(id=fid[c],column=c) for c in base_names]).to_csv(SUP/'feature_ids.csv',index=False)
trials=[]
for label,(_,relative) in RUNS.items():
    path=EXP/relative
    for q in path.rglob('validation_trials.csv'):
        a=csv(q);a.insert(0,'run',label);a.insert(1,'folder',str(q.parent.relative_to(path)));trials.append(a)
    for q in path.rglob('validation_trials.json'):
        a=js(q)
        if isinstance(a,list):
            a=pd.DataFrame(a);a.insert(0,'run',label);a.insert(1,'folder',str(q.parent.relative_to(path)));trials.append(a)
    for q in path.rglob('selection.json'):
        dest=SUP/'cards'/label/q.relative_to(path);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(q,dest);track(q)
    for q in list(path.glob('*dictionary*.csv'))+list(path.glob('*definitions*.csv')):
        shutil.copyfile(track(q),SUP/(label+'_'+q.name))
if trials:pd.concat(trials,ignore_index=True).to_csv(SUP/'all_validation_trials.csv',index=False)
W,H=landscape(A4);WIDTH=W-72;navy=colors.HexColor('#17354a');teal=colors.HexColor('#087f8c')
styles={n:ParagraphStyle(n,fontName=f,fontSize=s,leading=l,textColor=c,spaceAfter=a,splitLongWords=True)
 for n,f,s,l,c,a in [('title','Helvetica-Bold',23,27,navy,13),('head','Helvetica-Bold',13,17,teal,8),
 ('body','Helvetica',10,14,navy,8),('small','Helvetica',8.5,11,navy,6),('cell','Helvetica',8.2,10.5,navy,0),
 ('th','Helvetica-Bold',8.2,10.5,colors.white,0),('code','Courier',7.3,9.5,navy,0)]}
story=[]
def para(s,style='body'):
 value=str(s)
 if style!='code':
  value=re.sub(r'([;,])(?=\S)',r'\1 ',value)
  value=re.sub(r'(?<=[0-9])(?=(?:iterations|epochs|leaves|features|inputs|snapshots|channels|windows|quarters|rows|minutes|hours|finite|observations|timesteps|hidden|layers))',' ',value)
 return Paragraph(escape(value),styles[style])
def text(s,style='body'):story.append(para(s,style))
def page(title,subtitle=None):
 if story:story.append(PageBreak())
 text(title,'title')
 if subtitle:text(subtitle,'small')
def table(head,rows,widths=None,codecol=None):
 t=Table([[para(h,'th') for h in head]]+[[para(v,'code' if i==codecol else 'cell') for i,v in enumerate(row)] for row in rows],colWidths=widths or [WIDTH/len(head)]*len(head),repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eff4f6'),colors.white]),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
 story.append(t);story.append(Spacer(1,8))
def fmt(v):return '--' if pd.isna(v) else f'{v:.3f}'
def chunks(a,n):return [a[i:i+n] for i in range(0,len(a),n)]

page('BTC | 200-hour magnitude models','14 November 2026 | Six completed model stages | 17,873 matched origins per target | Log basis points')
text('Architectures, hyperparameters, features, formulas and ranked results','head')
text('Best means lowest pooled MAE among the saved forecasts audited here. The best observed UP and DOWN forecasts are research candidates, not independently confirmed deployment models. This report recomputes metrics from saved predictions; it does not train or run model inference.')
rows=[]
for target,label,forecast in [('UP','F','candidate'),('DOWN','E','candidate')]:
    r=pol[(pol.target==target)&(pol.run==label)&(pol.forecast==forecast)].iloc[0]
    rows.append([target,label+'.'+forecast,'Transformed nested HGB/linear ensemble' if target=='UP' else 'GRU/median plus paired HGB updates',fmt(r.MAE),fmt(r.RMSE),fmt(r.IC)])
table(['Target','Best observed','Model workflow','MAE','RMSE','Rank IC'],rows,[45,105,WIDTH-345,65,65,65])
text('UP F.candidate: 394.598 MAE. DOWN E.candidate: 378.915 MAE, identical to F.selected. Both beat mean and median MAE. Neither beats mean RMSE. These are quarterly forecast policies, not one fixed architecture or feature vector for the full period.')
table(['Baseline','UP MAE','UP RMSE','DOWN MAE','DOWN RMSE'],[[col.replace('training_','').title()]+[fmt(pol[(pol.target==t)&(pol.forecast==col)].iloc[0][m]) for t in ['UP','DOWN'] for m in ['MAE','RMSE']] for col in ['training_mean','training_median']],[180]+[(WIDTH-180)/4]*4)
text('Source scope: A initial F4Z search; B price follow-up; C long-feature search; D market HGB/MLP/GRU; E dual-loss refinement; F transformed dual refinement. The long-feature diagnostic run is an analysis of C, not an additional forecast model.')

page('Best-model status | observed scores versus selection')
table(['Workflow','MAE / RMSE','Interpretation'],[
 ['UP best observed: F.candidate','394.598 / 550.194','Lowest MAE among all full-coverage saved forecasts. Ungated candidate from the latest run.'],
 ['UP F.selected','397.814 / 551.022','Only October update passed this run-specific guards. Its fallback is E.candidate, which itself was diagnostic.'],
 ['UP original guarded: B.selected','411.963 / 570.677','Original HGB predictions; no new price features selected for UP. A.prediction and E.selected are identical.'],
 ['DOWN best observed: E.candidate','378.915 / 551.302','Lowest observed MAE. F.selected retains exactly these forecasts; all later DOWN updates were rejected.'],
 ['DOWN original guarded: D.selected','380.376 / 561.536','Earlier guarded GRU/median policy. Later E.selected slightly worsens to380.862 /561.675.'],
 ['RMSE winner among ranked forecasts','UP539.383 / DOWN525.554','The training-mean baseline has the lowest RMSE among the complete forecasts audited.']],[215,150,WIDTH-365])
text('A selected label describes the local selection rule, not the entire research history. In F, selected inherits E diagnostic candidates. Ranking F.selected alongside independently guarded policies without that distinction would be misleading.')
text('MAE improvements of best observed candidates: UP versus mean 2.60%, versus median 4.91%; DOWN versus mean 3.26%, versus median 0.80%. All values refer to repeatedly examined historical observations, with dependent overlapping 200-hour labels.')

page('Evaluation and score definitions')
table(['Item','Contract'],[
 ['Targets','BTCUSDT nonnegative path excursions over12,000minutes. UP(t)=max(0,max_{0<s<=200h}10000*ln(P(t+s)/P(t))); DOWN(t)=max(0,-min_{0<s<=200h}10000*ln(P(t+s)/P(t))). Canonical1minute path labels, valid zeros retained; not signed endpoint returns.'],
 ['Population',f"{data['A'].decision_time.min()} through {data['A'].decision_time.max()}. 17,873 matched origins per side;47,488eligible rows overall. 608development exclusions:304invalid path and304outside release."],
 ['Chronology','Jan/Apr/Jul/Oct2024 expanding fits;October retainedJanuary2025. Training label end strictly precedes boundary:origin+200h<boundary. Original48h input gap retained;200h maturity governs labels.'],
 ['MAE','mean(abs(y-p)); lower is better. Baselines are the quarterly training median and training mean.'],
 ['RMSE','sqrt(mean((y-p)^2)); lower is better. Large errors receive more weight than under MAE.'],
 ['Rank IC / bias','Spearman(y,p); higher rank correlation is better but not calibration. Bias=mean(p-y);negative means underprediction.'],
 ['Rankings','Complete saved policies, candidates and full-coverage fixed B variants. Identical predictions consolidated in full ranking; aliases preserved in CSV. Partial variants excluded from full-period comparisons.'],
 ['Audit','Six prediction hashes verified;target/time/issue/truth/baseline alignment checked. Pooled and period scores independently recomputed and matched to saved scores. Saved estimator hashes and metadata inspected; no new inference.'],
 ['Limitations','Repeated adaptive experimentation;not fresh confirmation or significance evidence. Earlier-selected features and architectures have been reused. No guaranteed future performance.']],[100,WIDTH-100])
for target in ['UP','DOWN']:
    page(target+' | policy rankings','MAE primary; RMSE and IC ranks separate. Selected rows in F inherit diagnostic candidates.')
    rr=pol[pol.target==target].to_dict('records')
    table(['MAE rank','Policy','Role','MAE','RMSE','IC','RMSE rank','Bias'],[[int(r['MAE_rank']),r['run']+'.'+r['forecast'],r['role'],fmt(r['MAE']),fmt(r['RMSE']),fmt(r['IC']),int(r['RMSE_rank']),fmt(r['Bias'])] for r in rr],[45,165,75,80,80,70,70,WIDTH-585])
for target in ['UP','DOWN']:
    for k,rr in enumerate(chunks(full[full.target==target].to_dict('records'),12)):
        page(target+f' | all full-coverage forecasts {k+1}','Fixed feature variants are post-selection diagnostics. Exact duplicate aliases are preserved in the support CSV.')
        table(['MAE rank','Forecast','Role','MAE','RMSE','IC','RMSE rank','IC rank'],[[int(r['MAE_rank']),r['run']+'.'+r['forecast'],r['role'],fmt(r['MAE']),fmt(r['RMSE']),fmt(r['IC']),int(r['RMSE_rank']),int(r['IC_rank'])] for r in rr],[45,235,100,65,65,60,60,WIDTH-630])

for target,label in [('UP','F'),('DOWN','E')]:
    page('Best '+target+' | quarterly scores','Best observed candidate; October model also servesJanuary2025. All errors in logbps.')
    g=data[label];g=g[g.target.str.startswith(target.lower())];rows=[]
    for period,a in g.groupby('period'):
        m=metrics(a.actual.to_numpy(),a.candidate.to_numpy());med=metrics(a.actual.to_numpy(),a.training_median.to_numpy());mn=metrics(a.actual.to_numpy(),a.training_mean.to_numpy())
        rows.append([period,len(a),fmt(m['MAE']),fmt(med['MAE']),fmt(mn['MAE']),fmt(m['RMSE']),fmt(mn['RMSE']),fmt(m['IC']),fmt(m['Bias'])])
    table(['Period','Rows','Model MAE','Median MAE','Mean MAE','Model RMSE','Mean RMSE','IC','Bias'],rows,[65,50,95,95,95,95,95,65,WIDTH-655])
    text('Pooled rankings do not imply improvement in every quarter. January2025 is scored separately but uses the October2024 fitted model.')

page('Best UP | nested architecture and exact blend formula','F.candidate uses A/B original HGB -> E dual update -> F transformed update. Each quarterly model is trained separately.')
text('Let A_q be the original UP HGB. Let L_q and M_q denote the E absolute-loss and squared-loss HGB predictions. Let L*_q and M*_q denote the F transformed-input predictions. All components are nonnegative and use training-only preprocessing.')
table(['Issue','E starting candidate C_q','Best UP final F_q','F update inputs'],[
 ['Jan','0.75*A +0.125*L +0.125*M','0.75*C +0.25*L*','Same5original inputs;quadratic HGB L1'],
 ['Apr','L (100% shallow L1)','0.75*C +0.0625*L* +0.1875*M*','12LONG inputs;raw-basis SGD/Ridge'],
 ['Jul','0.75*A +0.25*L','0.25*C +0.75*L*','Same5original inputs;quadratic HGB L1'],
 ['Oct / Jan2025','0.25*A +0.75*M','0.75*C +0.25*M*','Same3original inputs;quadratic HGB L2']],[100,205,245,WIDTH-550])
text('E uses shallow HGB in Jan/Apr/Oct and regularized HGB in Jul. F quadratic HGB uses60iterations/depth2/4leaves/leaf256. The April Ridge mean head minimizes squared error; the SGD head targets absolute error. Missing update inputs below80%finite revert to C or A at that update layer; these equations describe usable rows.')
text('Scores: MAE394.598115;RMSE550.194246. Exact tree thresholds, learned coefficients, missing-value statistics and neural weights are stored in the hashed joblib artifacts. Architecture alone cannot reproduce fitted numerical predictions.')

page('Best DOWN | nested architecture and exact blend formula','E.candidate: D selected GRU/median starting policy plus paired HGB updates; identical to F.selected.')
text('Let m_q be the quarterly training median. G8/G16 are bounded neural forecasts with38leverage inputs at60/40/20/0h. Let L/M be E shallow HGB absolute/squared-loss predictions.')
table(['Issue','D starting policy A_q','Best DOWN candidate C_q','HGB update pack'],[
 ['Jan','m','0.75*m +0.25*L','LONG+market69'],
 ['Apr','0.75*m +0.25*G8','0.5*A +0.5*L','Leverage38'],
 ['Jul','0.75*m +0.25*G8','0.5*A +0.375*L +0.125*M','LONG+market69'],
 ['Oct / Jan2025','0.9*m +0.1*G16','0.5*A +0.5*L','LONG+market69']],[105,205,245,WIDTH-555])
text('GRU8:single recurrent layer with8hidden units,linear scalar head. GRU16:16hidden units. Input shape4timesteps x76channels (38values+38missingflags per step). Output G=m+0.75*max(m,1)*tanh(head). When fewer than80% of required inputs are finite, the weighted neural adjustment is zero.')
text('E heads:depth2,4leaves,minleaf256,60iterations,rate0.05,L2=10,maxbins63;all mature training rows. Scores:MAE378.914575;RMSE551.301760. Updates failing input coverage fall back to A. The best DOWN forecast is a hybrid, not a pure standalone GRU.')

page('Best-model formulas | elementary predictors')
table(['Component','Formula and preprocessing'],[
 ['HGB','p=max(0,b0+eta*sum_k T_k(z)). T_k is a learned tree leaf update. Saved sklearn tree contributions already contain learning-rate scaling;do not multiply them again. MAE heads fit absolute_error;L2 heads fit squared_error.'],
 ['Original preprocessing','Replace missing values by training medians;append missing flags;center/scale expanded columns with training mean/std (tiny std=>1). Neural inputs additionally clip to[-8,8].'],
 ['GRU gates','r=sigmoid(W_ir*z+b_ir+W_hr*h_prev+b_hr);u=sigmoid(W_iu*z+b_iu+W_hu*h_prev+b_hu);n=tanh(W_in*z+b_in+r*(W_hn*h_prev+b_hn));h=(1-u)*n+u*h_prev. Initial hidden state0;last hidden enters a linear scalar head.'],
 ['Neural forecast','G=m+0.75*max(m,1)*tanh(w_h*h_last+b_h). Weighted policy=m+w*(G-m). Nonfinite-input coverage<80% gives median.'],
 ['Linear head','p=max(0,U*(b+sum_j beta_j*z_j)),U=max(training target median,1). Ridge minimizes squared error plus alpha*sum beta_j^2;SGD epsilon0 loss is absolute error with elastic-net penalty.'],
 ['Generic dual blend','p=(1-u)*anchor+u*((1-w)*L1+w*L2). u is update_weight;w is mean_weight. Zero head weights mean that head has no contribution.'],
 ['Exact numerical parameters','The companion model_components.json records estimator hyperparameters and node/pack identities. Full fitted weights and thresholds remain in the original hashed joblib files;no single universal coefficient vector applies to all quarters.']],[155,WIDTH-155])

page('Architectures and hyperparameters | A, B and C')
table(['Family','Settings and use'],[
 ['Original HGB','absolute_error;80iterations;depth3;7leaves;minleaf128;rate0.05;L2=1;maxbins255;seed48;early_stopping=False. Used by A/B and several C quarterly models.'],
 ['Ridge','alpha=0.01*n_training;train-only median imputation,missingflags,standardization. Candidate in A/C searches;the quarterly component appendix identifies saved Ridge winners.'],
 ['Screen tree','DecisionTreeRegressor absolute_error;depth2;minleaf128;seed48. Screening only,not an independently ranked full-period forecast. Screening subset capped4096 evenly spaced rows.'],
 ['A search','522maskedF4Z inputs;coverage>=80%,finite>=1000,variation;shortlist16,max6features. Empty-start forward/removal search for Ridge/HGB;3earlier purged quarters;0.1%medianMAEgain and2wins,elsemedian.'],
 ['B follow-up','Adds causal51price features in blocks/ratios,or removes old inputs. Selected changes require earlier gains versusoriginal. Fixed alternatives are saved diagnostics. UP retained original models in everyquarter.'],
 ['C fresh long search','116newfeatures:96LONGPRICE+20LONGMARKET. Shortlist20,max8;Ridge/HGB. Candidate must beat BOTHmean andmedian MAE by0.1% and2/3wins;else earlier-chosen mean/median. Market history features failed quality in this run.']],[140,WIDTH-140])
page('Architectures and hyperparameters | D market models')
table(['Configuration','Architecture / training'],[
 ['HGB','absolute_error;60iterations;depth3;7leaves;leaf128;rate0.05;L2=10;bins63;seed48;no early stopping. Estimator cap6000evenly spaced rows;full mature rows for preprocessing/median.'],
 ['MLP16','2d input channels(values+flags)->16ReLU->1linear head->bounded tanh residual. AdamW lr0.002,weightdecay0.1.'],
 ['MLP32/16','2d->32ReLU->16ReLU->1head->bounded tanh residual. lr0.001,decay0.03.'],
 ['GRU8 / GRU16','4timesteps at60/40/20/0h,2d channels per step;single GRU layer hidden8 or16->linear scalar->bounded tanh residual. GRU8 lr0.002/decay0.1;GRU16 lr0.001/decay0.03.'],
 ['Shared neural training','24epochs;batch512;AdamW;MAE;gradientnormclip1;seed48;deterministic;4096evenly spaced estimator rows. Full-training preprocessing/median. No targetlog.'],
 ['Packs and weights','Leverage38,flow/liquidation30,combined62. Independent target/quarter search over5configs and3packs;medianblend weights0.1/0.25/0.5/1.'],
 ['Gate','4earlier purged quarters;0.1%pooledmedianMAEgain;3/4wins/ties;no window>2%worse. Guarded selected fallbackmedian;lowest-validationMAE candidate separate.']],[140,WIDTH-140])
page('Architectures and hyperparameters | E and F updates')
table(['Component','Exact settings'],[
 ['E shallow HGB pair','absolute_error +squared_error;depth2;4leaves;minleaf256;60iterations;rate0.05;L2=10;maxbins63;seed48;no early stopping. All mature rows.'],
 ['E regularized pair','Same exceptdepth3,7leaves,minleaf128,80iterations.'],
 ['F linear pairs','raw,signed_log,rank,quadratic input bases. Ridge alpha0.01*n. SGD loss=epsilon_insensitive,epsilon0,elasticnet alpha0.001,l1ratio0.1,adaptive eta0.005,max1500iterations,tol1e-5,n_iter_no_change15,average=True,seed48,early_stopping=False.'],
 ['F HGB pair','Quadratic basis;two losses;E shallow settings. Constant target scale U=max(median,1),restored before scoring.'],
 ['Blend grid','mean_weight0/0.25/0.5/0.75/1;update_weight0.25/0.5/0.75/1. Three update packs:originalUPinputs/DOWNleverage38;12LONG;69LONG+market.'],
 ['Strict joint gate','Four earlier200h-purgedwindows. MAE must improve0.1%versus row-weighted smaller mean/medianMAE perwindow;RMSE improve0.1%versusmean;no pooled degradation versusrefittedanchor;3/4wins each;no>2%windowloss.'],
 ['Anchor distinction','E anchors originalB UP andD selectedDOWN. F anchors E candidates. Entire anchor stack refitted inside earlier windows;fallback preserves saved outer anchor. Feature/config/weight choices themselves remain adaptive historical research.']],[140,WIDTH-140])

page('Feature engineering | LONG hourly price formulas','Used by D/E/F. Distinct from the five-minute LONGPRICE universe in C.')
text('Let r_j be completed hourly log-bps returns formed from12five-minute endpoints in[t-1h,t). Define N_h=sum r;V_h=sqrt(sum r^2);A_h=sum abs(r);C=current cumulative hourly level;H_h/L_h=local high/low including start and end levels. Require complete windows;missing remainsNaN. Half-hour origins use the latest completed hour,never future filling.')
table(['Measurement','Formula'],[
 ['net / RMS volatility','N_h; V_h/sqrt(h)'],['up variance share','sum(max(r,0)^2)/max(V_h^2,1e-16)'],
 ['trend efficiency / normalized momentum','N_h/max(A_h,1e-8); N_h/max(V_h,1e-8)'],
 ['range / range position','H_h-L_h; (C-L_h)/max(H_h-L_h,1e-8)'],
 ['drawdown / rebound in volatility units','(H_h-C)/max(V_h,1e-8); (C-L_h)/max(V_h,1e-8)'],
 ['positive-hour share / jump share','fraction(r>0); max(r^2)/max(V_h^2,1e-16)'],
 ['hourly skew','Sample skewness of completed hourly returns, pandas rolling skew convention.'],
 ['vol_ratio_s / drift_change_s','(V_s/sqrt(s))/max(V_l/sqrt(l),1e-8); N_s/s-N_l/l.'],
 ['Generated lookbacks','24,100,168,200,336,400,720,1440hours;ratio/drift pairs24:200,200:400,168:720,336:1440.'],
 ['12-input update pack','RMS200/720; jump336; net168; range1440; upshare200/1440; drift200:400 and336:1440;range400;momentum200;rangeposition200.']],[210,WIDTH-210])
page('Feature engineering | NEWPRICE and LONGPRICE')
table(['Universe','Definition'],[
 ['B NEWPRICE','Five-minute returns strictly before origin;complete h-hour windows at1/4/12/24/48/168h. Eight per window:rv,up_rv,down_rv,net,past_range,distance_from_high,distance_from_low,max_abs_return. Plus3volatility ratios1:24,4:24,24:168 =51inputs.'],
 ['NEWPRICE equations','rv=sqrt(sum r^2);up_rv=sqrt(sum max(r,0)^2);down_rv=sqrt(sum min(r,0)^2);net=sum r;max_abs=max abs(r). Local cumulative past path gives range=H-L,drawdown=H-C,rally=C-L. Ratios=(rv_short/sqrt(short))/max(rv_long/sqrt(long),1e-6).'],
 ['C LONGPRICE','Uses FIVE-MINUTE returns at24/72/168/200/336/720/1440h.12measurements perwindow:net,rv,up_rv,down_rv,range,drawdown,rally,range_position,trend_efficiency,trend_to_vol,variance_balance,jump_share.'],
 ['C extra equations','range_position=(C-L)/max(H-L,1e-8);trend_efficiency=N/max(sum abs(r),1e-8);trend_to_vol=N/max(rv,1e-8);variance_balance=(upvariance-downvariance)/max(totalvariance,1e-8);jump_share=max(r^2)/max(sum r^2,1e-8).'],
 ['C cross-window features','6pairs24:200,72:200,168:720,200:720,336:1440,720:1440. vol_ratio=(rv_s/sqrt(s))/(rv_l/sqrt(l)),floor1e-8;trend_acceleration=N_s/s-(N_l-N_s)/(l-s). Note: this acceleration differs from LONG drift_change.'],
 ['C LONGMARKET','5masked4h percentile sources:basis,funding,flow,OI,liquidation severity. Mean/std over168/720h with80%expected half-hour coverage=>20columns;not admitted by the recorded quality screens.'],
 ['Missingness / meaning','Price windows require every5minute return finite. Historical paths are sampled canonical log-price paths,not intrabar highs/lows. No outcome or future label-validity flag is used as predictor.']],[145,WIDTH-145])
page('Feature engineering | market state and interactions')
table(['Group','Formula / definition'],[
 ['Source inputs','Eleven masked5m measurements:funding level/change,basis level/change,OI range position/unwind,flow pressure/activity,local/peer historical liquidation activity rates,venue breadth. They are historical measurements,not future liquidation amounts.'],
 ['STATE','For source s and past hhours:mean=average s(u),u in[t-h,t);deviation=s(t-30min)-mean;populationstd;high_share=fraction(s>=0.8). Min80%expected half-hour snapshots;no filling. Windows6/24/168/200h pluslag30min.'],
 ['funding_oi','mean(funding_level)*mean(oi_position)'],
 ['oi_lowflow','mean(oi_position)*(1-mean(flow_pressure))'],
 ['unwind_negative_momentum','mean(oi_unwind)*max(-LONG normalized_momentum,0)'],
 ['basis_funding','(mean(basis_level)-0.5)*(mean(funding_level)-0.5)'],
 ['liq_volatility / flow_momentum','mean(liq_history)*LONG RMS; (mean(flow_pressure)-0.5)*LONG normalized_momentum'],
 ['Focused leverage38','6pricecore +6funding/basis/OIfamilies x2stats(mean/deviation)x2windows(24/200h) +4interactions x2windows. Note oi_lowflow also depends on flow_pressure as an underlying source.'],
 ['Flow30 / combined62','6pricecore +5flow/liquidationfamilies x2stats x2windows +2interactions x2windows. Combined is the union of leverage andflow. Pricecore:RMS/momentum/rangeposition/upshare200h,drawdown168h,jump336h.'],
 ['GRU temporal inputs','Exact snapshots60/40/20/0h on full X inventory before label exclusions. Missing timestamps remain missing.38basefeatures become152orderedvalues plus152flags,arranged4x76.']],[170,WIDTH-170])
page('Feature engineering | transformed inputs')
table(['Transformation','Exact rule, fitted on training only'],[
 ['Impute / robust scale','Replace nonfinite values with training median m_x. v=(x-m_x)/IQR_x;IQR<=1e-8 becomes1.'],
 ['Raw','clip(v,-8,8). Raw means affine control,not unscaled inputs.'],
 ['Signed log','clip(sign(v)*ln(1+abs(v)),-8,8). Compresses extremes while retaining sign.'],
 ['Rank','2*(left insertion index+right insertion index)/(2*n_train)-1 using sorted finite training values. Midrank ties;out-of-range values map to extremes.'],
 ['Quadratic','Signed-log values plus each value squared;no full pairwise-product expansion. Append original missing flags.'],
 ['Final scaling','Center and scale basis columns using training mean/std (std<=1e-8 becomes1);clip[-8,8]. Store all parameters;prediction batches never refit transforms.'],
 ['Dimensions','Raw/log/rank:2d(value+flag). Quadratic:3d(value+square+flag). Neural D uses a different mean/std preprocessing implementation,not this robust basis.'],
 ['Target','Positive constant U=max(training target median,1) only. No second log transform,rank transform,outcome clipping or tail trimming. Scores are restored original logbps.']],[160,WIDTH-160])
text('F4Z upstream percentiles and robust Z-scores are consumed as supplied. Their exact upstream construction is not recomputed here;the report preserves source names and timeframes rather than inventing unavailable upstream formulas.')

# Best-model exact component/feature identities, with all nested dependencies.
for target,label in [('UP','F'),('DOWN','E')]:
    selected_roots=[r for r in roots if r['run']==label and r['target']==target and r['forecast']=='candidate']
    for r in selected_roots:
        page(f'Best {target} | {r["issue"]} exact components','Node IDs below define the executable blend graph. Pack IDs expand to exact feature IDs later. Zero-weight heads are recorded but do not contribute on usable rows.')
        rr=[nodes[int(i[1:])-1] for i in r['nodes']]
        table(['Node','Architecture','Pack','Prediction formula'],[[n['id'],n['kind'],n['pack'],n['formula']] for n in rr],[50,180,55,WIDTH-285])
        text('Final forecast node: '+r['root']+'. All dependency inputs,including missing-input fallbacks,are included. Full estimator settings are exported in model_components.json.')
        for n in rr:
            if n['kind']=='Blend':continue
            setting=n['settings']
            if isinstance(setting,dict):
                pp=setting.get('params',setting)
                keys=['loss','alpha','max_iter','max_depth','max_leaf_nodes','min_samples_leaf','learning_rate','l2_regularization','hidden','lr','decay','weight','median']
                compact='; '.join(k+'='+str(pp[k]) for k in keys if k in pp)
                if 'transform' in setting:compact+='; transform='+str(setting['transform'])+'; target unit='+str(setting['unit'])
                text(n['id']+': '+compact,'small')
for label,(name,_) in RUNS.items():
    rr=[r for r in roots if r['run']==label]
    for k,part in enumerate(chunks(rr,8)):
        page(label+' | '+name+f' / quarterly models {k+1}','Exact saved-model roots and feature packs;October also servesJanuary2025. Full component graph and fitted estimator hyperparameters are in the support archive.')
        table(['Side','Issue','Forecast','Root architecture','Node','Input pack'],[[r['target'],r['issue'],r['forecast'],nodes[int(r['root'][1:])-1]['kind'],r['root'],r['pack']] for r in part],[50,90,100,240,70,WIDTH-550])

# Consolidated packs printed using IDs; exact full names follow.
packrows=[]
for pid,cols in features.items():
    bases=list(dict.fromkeys(re.sub(r'^SEQ[0-9]+__','',c) for c in cols))
    seq=any(c.startswith('SEQ') for c in cols)
    # Mixed packs contain sequence anchor plus current correction values.
    current=[fid[c] for c in cols if not c.startswith('SEQ')]
    seqbase=list(dict.fromkeys(fid[re.sub(r'^SEQ[0-9]+__','',c)] for c in cols if c.startswith('SEQ')))
    desc=('Sequence60/40/20/0h: '+', '.join(seqbase)+'; ' if seq else '')+('Current: '+', '.join(current) if current else ('No inputs' if not seq else ''))
    packrows.append([pid,len(cols),desc])
for k,part in enumerate(chunks(packrows,6)):
    page(f'Exact feature packs | {k+1}','Pack lists preserve model input order. Sequence base IDs repeat at each lag. Preprocessing appends missing flags. Blend packs include fallback dependencies.')
    table(['Pack','Raw values','Ordered feature IDs'],part,[50,65,WIDTH-115])
for k,part in enumerate(chunks(base_names,18)):
    page(f'Exact feature-name index | {k+1}','F4Z is inherited source data;LONGPRICE is5minute engineering;LONG is hourly engineering;STATE/INTERACT follow the formulas above.')
    table(['ID','Exact column name'],[[fid[c],c] for c in part],[50,WIDTH-50],codecol=1)
for k,part in enumerate(chunks(partial,20)):
    page(f'Partial fixed variants | {k+1}','Excluded from the full-period ranking. Per-period and available-row scores remain in all_forecasts_period_metrics.csv.')
    table(['Side','Stage','Variant','Available rows'],[[r['target'],r['run'],r['forecast'],r['rows']] for r in part],[60,60,WIDTH-220,100])
page('How to read feature importance and scores')
text('A feature appearing in a selected pack does not establish independent predictive value or causal importance. Correlated inputs can substitute for each other;neural and tree interactions mean feature contributions are not additive.')
text('Validation-trial scores and model settings are preserved in all_validation_trials.csv and original selection cards. Inner-only configurations are not assigned fabricated evaluation ranks. Fixed full-coverage B variants are ranked as diagnostics;partial variants are not compared against full-period policies.')
text('Earlier candidate diagnostics found the highest10%actual moves contributed62.65%UP and67.30%DOWN of E candidate squared error. Average E candidate errors were-170.77UP and-160.86DOWN logbps. These are hindsight descriptions;the report does not fit a correction from evaluation outcomes.')
text('The mean baseline remains the best RMSE forecast among this audited complete saved forecasts. Best observed model MAE and best RMSE can belong to different forecasts. A single model has not established superiority on all three criteria:mean MAE,median MAE,andmean RMSE.')
for k,part in enumerate(chunks(audit,3)):
    page(f'Provenance and reproducibility | {k+1}','Paths are relative to Week2/Code/experiments. Full SHA256 values and input/code contracts are in the companion archive.')
    table(['ID','Experiment','Source path'],[[r['run'],r['name'],r['path']] for r in part],[35,160,WIDTH-195])
    text('Source prediction hashes,estimator hashes and score arithmetic were checked. Replay flags are taken from completed runs;this report performs metadata inspection only. Original datasets and experiment folders are unchanged.')
    text('Companion:BTC_200h_Report_Support_2026-11-14.zip contains rankings,period metrics,exact feature IDs/packs,quarterly roots,component hyperparameters,selection cards,validation trials,contracts,and audit hashes. Numerical learned tree/neural parameters remain in the source joblib files.')

def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#c9d6df'));canvas.line(36,32,W-36,32)
    canvas.setFont('Helvetica',8);canvas.setFillColor(navy);canvas.drawString(36,20,'BTC 200h | Saved research evidence | 14 Nov 2026');canvas.drawRightString(W-36,20,str(doc.page))
doc=SimpleDocTemplate(str(PDF),pagesize=landscape(A4),leftMargin=36,rightMargin=36,topMargin=30,bottomMargin=43,title='BTC 200h models, features and rankings',author='Research report')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(PDF);texts=[p.extract_text() for p in reader.pages]
assert all(len(t)>100 for t in texts)
assert '394.598' in ''.join(texts) and '378.915' in ''.join(texts)
(SUP/'source_audit.json').write_text(json.dumps(dict(runs=audit,files=hashes,models=model_hashes),indent=2))
(SUP/'REPORT_CHECKS.json').write_text(json.dumps(dict(pages=len(texts),prediction_hashes_verified=6,model_hashes_verified=len(model_hashes),origins_per_target=17873,metrics_recomputed=True,no_training=True,no_inference=True,feature_count=len(base_names),packs=len(packs)),indent=2))
zip_path=OUT/'BTC_200h_Report_Support_2026-11-14.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for q in SUP.rglob('*'):
        if q.is_file():z.write(q,q.relative_to(SUP))
print(json.dumps(dict(pdf=str(PDF),pages=len(texts),support=str(zip_path),full_rank_counts=full.groupby('target').size().to_dict(),features=len(base_names),packs=len(packs))))
print(full.groupby('target').head(3).to_string(index=False))
