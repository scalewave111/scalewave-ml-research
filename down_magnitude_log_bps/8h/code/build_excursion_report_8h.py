"""Saved-results report only: no model fitting or inference."""
from pathlib import Path
import json, hashlib, shutil
import numpy as np
import pandas as pd
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'Code/experiments/excursion_pipeline_8h_v1/20261113T104458041903Z'
SEARCH=RUN/'search/20261113T104505694341Z'
FOLLOW=RUN/'price_followup/20261113T110455348677Z'
TUNE=ROOT/'Code/experiments/excursion_tuning_8h_v1/20261113T112747680612Z'
OUT=ROOT/'Report/output/pdf/ScaleWave_8h_Excursion_Models_Report.pdf'
SUP=ROOT/'Report/eight_hour_excursion_report_sources'
SUP.mkdir(parents=True,exist_ok=True);OUT.parent.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={}
def readj(p):
    manifest[str(p)]=sha(p)
    return json.loads(p.read_text())
def readcsv(p):
    manifest[str(p)]=sha(p)
    return pd.read_csv(p)
frames=[]
for d in [SEARCH,FOLLOW,TUNE]:
    done=readj(d/'COMPLETE.json');assert done['status']=='COMPLETE'
    assert sha(d/'predictions.csv')==done['predictions_sha256']
    f=readcsv(d/'predictions.csv').sort_values(['target','decision_time']).reset_index(drop=True)
    assert len(f)==36962 and not f.duplicated(['target','decision_time']).any()
    frames.append(f)
for f in frames[1:]:
    assert f[['target','decision_time','issue']].equals(frames[0][['target','decision_time','issue']])
    np.testing.assert_allclose(f.actual,frames[0].actual,rtol=0,atol=1e-10)
np.testing.assert_allclose(frames[0].prediction,frames[1].original,rtol=0,atol=1e-7)
np.testing.assert_allclose(frames[1].selected,frames[2].current,rtol=0,atol=1e-10)
m=readcsv(FOLLOW/'metrics_all_variants.csv');tm=readcsv(TUNE/'metrics.csv')
initial=readj(SEARCH/'selections.json');follow=readj(FOLLOW/'selections.json')
contracts=[readj(d/'CONTRACT.json') for d in [SEARCH,FOLLOW,TUNE]]
plans={}
for d in sorted(TUNE.glob('2024-*')):
    plans[d.name]=readj(d/'selection.json')
original=sorted({c for s in initial+follow for c in s['features'] if c.startswith('F4Z__')})
ids={c:f'F{i+1:02d}' for i,c in enumerate(original)}
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='B',fontName='Helvetica',fontSize=9.5,leading=13,spaceAfter=8,textColor=colors.HexColor('#25364a')))
styles.add(ParagraphStyle(name='C',fontName='Helvetica',fontSize=7.8,leading=10))
styles.add(ParagraphStyle(name='S',fontName='Helvetica',fontSize=8,leading=11,spaceAfter=6))
story=[]; section=0
p=lambda t,s='B':Paragraph(t,styles[s])
def title(t):
    global section
    if story:story.append(PageBreak())
    section+=1
    story.extend([p('SCALEWAVE / BTCUSDT / 8-HOUR EXCURSIONS','S'),p(f'{section:02d}  {t}','Title'),Spacer(1,10)])
def para(t):story.append(p(t))
def table(headers,rows,widths):
    data=[[p(escape(str(v)),'C') for v in row] for row in [headers]+rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce9f4')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f3f6f9')])]))
    story.extend([t,Spacer(1,9)])
side=lambda t:'UP' if t.startswith('up_') else 'DOWN'
num=lambda v:f'{v:.4f}'
labels={'selected':'Price follow-up: selected','original':'Initial F4Z-selected model','training_mean':'Training mean baseline','training_median':'Training median baseline','all_price_plus_original':'Original + all price features','price_only':'Price features only','mae_selected':'Tuned MAE strategy','rmse_selected':'Tuned RMSE strategy','current':'Price follow-up: selected'}
def label(s):return labels.get(s,s.replace('add_','Add ').replace('remove_','Remove ').replace('price_','price ').replace('_',' '))
def block(cols):
    news=[c for c in cols if c.startswith('NEWPRICE')]
    hours=[h for h in [1,4,12,24,48,168] if f'NEWPRICE__rv__{h}h' in news]
    ratios=any('rv_ratio' in c for c in news)
    return ('8 measures at '+','.join(str(h)+'h' for h in hours) if hours else '')+('; 3 ratios' if ratios else '') if news else 'None'
def membership(cols):return ', '.join(ids[c] for c in cols if c in ids)

title('Results and model inventory')
para('Prepared 13 November 2026. Covers all three completed stages of the <b>8-hour canonical UP/DOWN excursion</b> workflow: original feature search, price-feature follow-up and model tuning. Separate UP and DOWN models use every valid origin, including zero excursions.')
table(['Side','Strategy','MAE','RMSE','Rank IC'],[[side(r.target),label(r.model),num(r.mae),num(r.rmse),num(r.ic)] for _,r in tm[(tm.period=='ALL')&tm.model.isin(['current','mae_selected','rmse_selected'])].iterrows()],[40,190,90,90,90])
para('Tuning reduced MAE by <b>0.15% UP / 0.28% DOWN</b> relative to the price follow-up. The RMSE strategy reduced RMSE by <b>3.68% UP / 3.56% DOWN</b>, with higher pooled MAE. All errors are log basis points. Scores pool four quarterly models; they are not scores of the October model alone.')
table(['Family / role','Use in this experiment'],[['Histogram gradient boosting','Final UP/DOWN models; absolute-error and squared-error versions; some 50/50 blends with prior boosting models.'],['Ridge regression','Initial feature-search candidate, squared-error fit and MAE selection.'],['Single decision tree','Depth-2 feature screening, not the final boosting model.'],['Training mean / median','Constant within each quarterly fit; comparison baselines and initial median fallback.']],[155,345])
para('Rankings below include full-coverage saved strategies only. Initial screening and unselected tuning candidates have validation scores, not full evaluation predictions. Historical evaluation has been repeatedly exposed; ranking does not establish independent confirmation.')

title('Targets, dates and evaluation metrics')
para('<b>UP = 10,000 ln(Pmax/P0).</b> <b>DOWN = 10,000 ln(P0/Pmin).</b> Pmax/Pmin are the highest/lowest canonical one-minute reference prices from origin through 480 minutes later, including origin. Both labels are nonnegative. These are excursions from origin, not endpoint returns or running-peak drawdowns.')
table(['Item','Definition'],[['Population','29,615 initial origins + 18,481 development origins; original half-hour sampling and 48h boundary gap preserved.'],['Evaluation range',f'{frames[2].decision_time.min()} to {frames[2].decision_time.max()}'],['Quarterly refits','1 Jan / 1 Apr / 1 Jul / 1 Oct 2024; expanding mature history. October also predicts January 2025, separately reported.'],['Maturity rule','Training origin + 8h strictly before fit/validation boundary. Validation labels end strictly before each window end.'],['Earlier validation','Three 3-month windows, starting 9, 6 and 3 months before each quarterly fit.'],['MAE','mean(abs(actual - prediction)); lower is better; main feature-search objective.'],['RMSE / MSE','sqrt(mean((actual - prediction)^2)); MSE is the unrooted value used for selection. Emphasizes large errors.'],['Rank IC','Spearman correlation between actual and predicted excursions; higher is better. Undefined for a constant series.']],[135,365])
para('Quarter-specific baselines are constant within a fit but can change between quarters, so pooled baseline IC need not be zero or undefined. Log bps are not ordinary percentage returns; no unit conversion is applied to these metrics.')

title('Architecture and hyperparameters')
para('<b>Original / price-follow-up path:</b> raw features -&gt; training-median imputation -&gt; append one missing indicator per raw feature -&gt; training-only centering/scaling -&gt; additive boosted trees -&gt; max(0, prediction). Thus N raw inputs become 2N inputs.')
para('<b>New tuning path:</b> raw features -&gt; training-fitted histogram bins with native missing-value routing -&gt; additive boosted trees -&gt; max(0, prediction). No separate scaling or missing indicators. A selected blend is 0.5 x current prediction + 0.5 x new prediction. No neural network or residual learner is used.')
table(['Setting','Original HGB','Tune S0','Tune S1','Tune S2'],[['Iterations',80,120,160,160],['Learning rate','.05','.04','.04','.04'],['Maximum depth',3,2,3,4],['Maximum leaves',7,4,7,12],['Minimum samples / leaf',128,128,64,128],['L2 regularization',1,10,5,10],['Maximum histogram bins','255 (default)',127,127,127],['Loss','Absolute error','Absolute / squared','Absolute / squared','Absolute / squared'],['Random seed',48,48,48,48],['Early stopping','False','False','False','False']],[140,90,90,90,90])
para('Absolute-error models target conditional medians; squared-error models target conditional means. Iterations add small regression trees to an initial constant prediction. Depth/leaves are maximum limits, not guarantees that every tree uses all available nodes. Execution uses two threads.')
para('<b>Initial search controls:</b> Ridge alpha = 0.01 x training rows. Screening tree: absolute_error, depth 2, minimum leaf 128, seed 48; at most 4,096 evenly spaced training rows for screening. Final boosting fits use all mature training rows.')

title('How features and models were selected')
table(['Stage','Selection process'],[['1. Original F4Z search','522 masked value columns; no mandatory features. Every inner training window requires >=80% finite coverage, >=1,000 readings and variation. Single-feature tree screen; earliest-training Spearman redundancy threshold 0.90; shortlist up to 16. Ridge and boosting forward search up to 6 features plus removal checks. MAE gain >=0.1% and wins in >=2/3 windows; median fallback.'],['2. Price follow-up','Retains each original selected feature set as anchor. Tests up to 2 greedy block additions, original+all 51 price, price-only and group removals. Alternative must beat original and median by >=0.1% pooled MAE and >=2/3 window wins against each. Otherwise retains original.'],['3. Tuning','Three packs: current selected, current+all 51 price, price-only. Two losses x three configurations = 18 candidate fits per side/cycle/window. Each also tested as a 50/50 blend with current.'],['Tuned MAE rule','>=0.1% pooled MAE gain versus current, >=2/3 window wins and pooled MSE no worse. Otherwise retains current.'],['Tuned RMSE rule','>=0.1% pooled MSE gain versus current and >=2/3 window wins. No MAE guard; may sacrifice MAE. Otherwise retains current.']],[125,375])
para('Choices were saved before outer fitting/evaluation. Outer metrics did not drive these selectors. Prior feature sets had already used the same earlier validation periods; the new tuning is adaptive research, not fresh nested validation. A validation guard does not guarantee later gains.')

title('Construction of the 51 price features')
para('Input: canonical five-minute BTC log returns r in log bps. At forecast origin t, use return endpoints in [t-h, t), strictly before t. A complete h-hour window requires 12h returns. Missing/incomplete windows stay missing. Reconstruct L = [0, cumulative sum(r)] within the window.')
table(['Measure / exact token','Formula or definition'],[['Realized volatility / rv','sqrt(sum(r^2))'],['Upward volatility / up_rv','sqrt(sum(max(r,0)^2))'],['Downward volatility / down_rv','sqrt(sum(min(r,0)^2))'],['Net movement / net','sum(r)'],['Range / past_range','max(L) - min(L)'],['Distance from high / distance_from_high','max(L) - L_last'],['Distance from low / distance_from_low','L_last - min(L)'],['Largest return / max_abs_return','max(abs(r))']],[225,275])
para('Apply all eight measures at <b>1, 4, 12, 24, 48 and 168 hours</b>: 48 inputs. Exact names: <b>NEWPRICE__{token}__{h}h</b>. These historical lookbacks are independent of the 8-hour forecast horizon. No new 8-hour lookback was added in tuning.')
para('Add three normalized ratios: (RV_short / sqrt(short hours)) / max(RV_long / sqrt(long hours), 0.000001), with pairs 1h/24h, 4h/24h, 24h/168h. Exact names: NEWPRICE__rv_ratio__1h_24h, NEWPRICE__rv_ratio__4h_24h, NEWPRICE__rv_ratio__24h_168h.')
para('Illustration using a complete one-hour window with returns [10,-5,0,0,0,0,0,0,0,0,0,0]: RV=11.1803, up RV=10, down RV=5, net=5. Path levels span 0 to10 and end at5: range=10, distance from high=5, distance from low=5, largest absolute return=10 log bps.')
para('Range/high/low features use five-minute close paths, not intrabar exchange extremes. Prefix and future-perturbation tests checked causality during runner preparation.')

ranked=[]
for target in ['up_magnitude_log_bps','down_magnitude_log_bps']:
    title(side(target)+' model ranking')
    r=m[(m.target==target)&(m.period=='ALL')&m.full_period_coverage.eq(True)&m.rows.eq(18481)].copy()
    r=pd.concat([r,tm[(tm.target==target)&(tm.period=='ALL')&tm.model.isin(['mae_selected','rmse_selected'])]],ignore_index=True)
    for col,asc in [('mae',True),('rmse',True),('ic',False)]:r[col+'_rank']=r[col].rank(method='min',ascending=asc).astype('Int64')
    r=r.sort_values(['mae','model']);ranked.append(r)
    para('Primary order: MAE, lower is better. Separate RMSE and IC ranks are shown. Every row covers the same 18,481 origins and canonical labels. Initial-search predictions duplicate the follow-up original; tuning current duplicates follow-up selected, so these are listed once.')
    table(['MAE rank','Model / strategy','MAE','RMSE (rank)','IC (rank)'],[[row.mae_rank,label(row.model),num(row.mae),f'{row.rmse:.3f} ({row.rmse_rank})',f'{row.ic:.3f} ({row.ic_rank})'] for _,row in r.iterrows()],[42,211,77,90,80])
    para('Only the rows labelled selected/tuned represent earlier-validation selection policies. Other feature variants are diagnostic comparisons; their ranking is post-hoc. Adaptive variants with partial coverage are excluded and exported separately. No full-data rank is assigned to unselected tuning candidates or single-feature screen models.')
pd.concat(ranked).to_csv(SUP/'ranked_full_coverage_models.csv',index=False)
m[(m.period=='ALL')&~m.full_period_coverage.eq(True)].to_csv(SUP/'partial_coverage_variants.csv',index=False)

for target in ['up_magnitude_log_bps','down_magnitude_log_bps']:
    title(side(target)+' evaluation by period')
    table(['Period','Strategy','Rows','MAE','RMSE','IC'],[[r.period,label(r.model),r.rows,num(r.mae),num(r.rmse),num(r.ic)] for _,r in tm[(tm.target==target)&tm.model.isin(['current','mae_selected','rmse_selected'])].iterrows()],[58,170,45,76,76,75])
    para('UP tuned-MAE: unchanged Q1/Q4/Jan, slightly worse MAE in Q2, better in Q3. DOWN tuned-MAE: better MAE Q1-Q3/Jan, worse Q4. RMSE-selected improves RMSE in every period for both sides but has higher pooled MAE. These are saved-table comparisons, not significance tests.')

for start in range(0,len(original),16):
    title('Exact original features'+(' (continued)' if start else ''))
    para('IDs abbreviate the exact F4Z columns used in the quarterly models. The full source names preserve family, measurement, lookback and value field. RZ, percentile and historical active rate are distinct source measurements; historical active rate describes past peer activity frequency, not current liquidation amount.')
    table(['ID','Exact feature name'],[[ids[c],c] for c in original[start:start+16]],[40,460])

for stage,entries in [('Initial search',initial),('Price follow-up',follow)]:
    title(stage+': features by cycle')
    table(['Fit','Side','Raw count','Original IDs','Price blocks'],[[s['issue'][:7],side(s['target']),len(s['features']),membership(s['features']),block(s['features'])] for s in entries],[60,40,45,175,180])
    para('All selected initial and follow-up models in these saved runs use absolute-error histogram boosting. Raw counts exclude missing indicators. The earlier dictionary plus the exact price-name template specifies every column in each row. January 2025 reuses October.')
    (SUP/(stage.lower().replace(' ','_')+'_selections.json')).write_text(json.dumps(entries,indent=2))

members=[]
for policy in ['mae_selected','rmse_selected']:
    title(label(policy)+': architecture and features')
    rows=[]
    for key,plan in plans.items():
        issue=key[:7];target=key[8:];choice=plan['choices'][policy]
        anchor=next(s for s in follow if s['issue'][:7]==issue and s['target']==target)
        if choice is None:
            cols=anchor['features'];arch='Original HGB retained';blend='100% prior'
        else:
            spec=plan['specs'][choice['candidate']];cols=spec['features']
            arch=('L1' if spec['loss']=='absolute_error' else 'L2')+' / S'+str(spec['setting'])
            blend='50% prior + 50% new' if choice['weight']==.5 else '100% new'
        rows.append([issue,side(target),arch,blend,len(cols),membership(cols),block(cols)])
        members.append(dict(policy=policy,issue=issue,target=target,architecture=arch,blend=blend,new_features=cols,prior_features=anchor['features'],union_features=sorted(set(cols+anchor['features'])) if choice and choice['weight']==.5 else cols))
    table(['Fit','Side','Model','Blend','N','Original IDs','Price blocks'],rows,[51,42,62,69,25,111,140])
    para('L1 = absolute-error loss; L2 = squared-error loss. S0/S1/S2 refer to the hyperparameter table. N counts raw features in the new component (or retained component). For blends, the prior component keeps its own features and preprocessing listed on the price-follow-up page; N is not the union count. Full component and union lists are saved in tuning_feature_membership.json.')
    para('Latest October MAE strategy: UP retains 13 features and original 80-iteration model; DOWN uses 55 features and tuned S1. Latest October RMSE strategy: UP uses 13 features/S1/L2; DOWN uses 55 features/S0/L2. These latest models also predict January 2025.')
(SUP/'tuning_feature_membership.json').write_text(json.dumps(members,indent=2))

title('Evidence, reproducibility and scope')
table(['Stage','Saved run identifier'],[['Initial search','excursion_pipeline_8h_v1/20261113T104458041903Z/search/20261113T104505694341Z'],['Price follow-up','excursion_pipeline_8h_v1/20261113T104458041903Z/price_followup/20261113T110455348677Z'],['Tuning','excursion_tuning_8h_v1/20261113T112747680612Z']],[105,395])
para('All paths above are under Code/experiments/. Evidence: COMPLETE.json, CONTRACT.json, predictions.csv, selections.json and per-cycle selection.json; follow-up metrics_all_variants.csv and tuning metrics.csv. Saved source snapshots define architecture, preprocessing and feature formulas.')
para('Report verification: all three prediction SHA256 values match completion records; exact target/timestamp/issuance alignment and actual labels agree across stages. Initial predictions match the follow-up original; follow-up selected predictions match tuning current. Run records report successful model replay. Metrics were read from saved tables, not independently recomputed. No training or model inference was performed for this report.')
para('Supporting folder: Report/eight_hour_excursion_report_sources/. Contains full-precision rankings, partial-coverage comparisons, exact feature membership, metrics, selection cards and source hashes. Unselected tuning trials are validation-only; detailed trials remain in the source run.')
para('History and earlier validation were adaptively reused. Overlapping 8h labels reduce independence. Original X availability and provisional-source limitations are inherited. No uncertainty intervals, statistical significance, fresh-data confirmation, trading-cost evaluation or profitability claim is supplied. Feature membership is not a feature-importance ranking.')
for d in [SEARCH,FOLLOW,TUNE]:
    for path in d.glob('*.py'):manifest[str(path)]=sha(path)
for name,df in [('followup_metrics',m),('tuning_metrics',tm)]:df.to_csv(SUP/(name+'.csv'),index=False)
(SUP/'source_hashes.json').write_text(json.dumps(manifest,indent=2))
(SUP/'tuning_selections.json').write_text(json.dumps(plans,indent=2))
def footer(canvas,doc):
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#607b95'))
    canvas.drawString(42,24,'ScaleWave | 8-hour excursions | 13 Nov 2026');canvas.drawRightString(553,24,str(doc.page))
SimpleDocTemplate(str(OUT),pagesize=(595,842),leftMargin=42,rightMargin=42,topMargin=35,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
