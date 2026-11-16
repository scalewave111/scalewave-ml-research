"""Read-only experiment audit and standalone HTML research report; no fitting."""
from pathlib import Path
import hashlib
import html
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
EXP=ROOT/'Code/experiments'
OUT=ROOT/'Deliverables/report_72h_20261114'
RUNS=[
 ('Price feature follow-up','excursion_pipeline_72h_v1/20261113T111815030467Z/price_followup/20261113T113500606206Z'),
 ('Tuned original trees','excursion_tuning_72h_v1/20261113T120003142401Z'),
 ('Rolling median policy','rolling_medians_72h_v1/20261113T121450890192Z'),
 ('Original neural MLP','excursion_neural_72h_v1/20261113T123304519186Z'),
 ('Fresh long-feature search','long_features_72h_v1/20261114T074315743848Z'),
 ('Compact trees','compact_trees_72h_v1/20261114T080702116226Z'),
 ('Transformed spline and ExtraTrees','transformed_models_72h_v1/20261114T085824921737Z'),
 ('Guarded residual and sizing models','guarded_down_72h_v1/20261114T112121591551Z'),
 ('Quarterly anchor residual tree','quarterly_anchor_down_72h_v1/20261114T113416293003Z'),
 ('Fast market HGB','market_state_down_72h_fast_v1/20261114T120044967681Z'),
 ('Focused market HGB','focused_hgb_both_72h_v1/20261114T121742176686Z'),
 ('Best-model residual calibration','best_models_calibration_72h_v1/20261114T122359645306Z'),
 ('Market MLP and GRU search','neural_market_down_72h_v1/20261114T123111032438Z'),
 ('UP market GRU search','gru_market_up_72h_v1/20261114T123538599800Z')]

def esc(v):return html.escape(str(v))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(rows,columns=None):
    frame=pd.DataFrame(rows,columns=columns)
    return frame.to_html(index=False,border=0,float_format=lambda v:f'{v:.4f}',escape=True,classes='data')
def section(title,body):return f'<section><h2>{esc(title)}</h2>{body}</section>'
def para(s):return '<p>'+s+'</p>'
def detail(title,obj):return f'<details><summary>{esc(title)}</summary><pre>{esc(json.dumps(obj,indent=2,default=str))}</pre></details>'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    reference=pd.read_csv(EXP/RUNS[3][1]/'predictions.csv').sort_values(['target','decision_time'])
    rows=[];periods=[];audit=[];data={};selections={};contracts={}
    for name,relative in RUNS:
        path=EXP/relative
        complete=json.loads((path/'COMPLETE.json').read_text())
        assert complete['status']=='COMPLETE' and sha(path/'predictions.csv')==complete['predictions_sha256'],name
        df=pd.read_csv(path/'predictions.csv').sort_values(['target','decision_time'])
        if 'selected' not in df and 'new_features' in df:
            df['selected']=df['new_features']
        assert not df.duplicated(['target','decision_time']).any()
        data[name]=df
        selections[name]=json.loads((path/'selections.json').read_text())
        contracts[name]=json.loads((path/'CONTRACT.json').read_text())
        for target,part in df.groupby('target'):
            ref=reference[reference.target.eq(target)]
            assert len(part)==18385 and part.decision_time.tolist()==ref.decision_time.tolist(),name
            np.testing.assert_allclose(part.actual,ref.actual,rtol=0,atol=1e-10)
            variants=['selected']
            if 'candidate' in part and not np.allclose(part.selected,part.candidate,rtol=0,atol=1e-10):variants.append('candidate')
            for variant in variants:
                pred=part[variant].to_numpy();truth=part.actual.to_numpy()
                assert np.isfinite(pred).all()
                mae=float(np.abs(truth-pred).mean());rmse=float(np.sqrt(np.mean((truth-pred)**2)))
                ic=float(pd.Series(truth).corr(pd.Series(pred),method='spearman'))
                baseline=float(np.abs(ref.actual-ref.training_median).mean())
                rows.append(dict(target=target.split('_')[0].upper(),experiment=name,forecast=variant,MAE=mae,RMSE=rmse,Spearman=ic,
                                 gain_percent=100*(baseline-mae)/baseline,rows=len(part),source=relative))
                for period,sub in part.groupby('period'):
                    periods.append(dict(target=target.split('_')[0].upper(),experiment=name,forecast=variant,period=period,
                        MAE=float(np.abs(sub.actual-sub[variant]).mean()),RMSE=float(np.sqrt(np.square(sub.actual-sub[variant]).mean()))))
        audit.append(dict(experiment=name,path=relative,prediction_sha256=complete['predictions_sha256'],rows=len(df),hash_verified=True,
                          reported_replay=complete.get('replay_passed',complete.get('prefix_replay_passed'))))
    for target,part in reference.groupby('target'):
        baseline=float(abs(part.actual-part.training_median).mean())
        for col,label in [('training_median','Quarterly median baseline'),('training_mean','Quarterly mean baseline')]:
            rows.append(dict(target=target.split('_')[0].upper(),experiment=label,forecast='baseline',MAE=float(abs(part.actual-part[col]).mean()),
                RMSE=float(np.sqrt(np.square(part.actual-part[col]).mean())),Spearman=float(part.actual.corr(part[col],method='spearman')),
                gain_percent=100*(baseline-float(abs(part.actual-part[col]).mean()))/baseline,rows=len(part),source=RUNS[3][1]))
    ranking=pd.DataFrame(rows).sort_values(['target','MAE','experiment']).reset_index(drop=True)
    ranking['rank']=ranking.groupby('target').cumcount()+1
    ranking.to_csv(OUT/'model_rankings.csv',index=False)
    pd.DataFrame(periods).to_csv(OUT/'period_metrics.csv',index=False)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    sections=[]
    sections.append(section('Findings',para('The best observed 72-hour UP forecast is the original regularized MLP, with MAE <b>234.3441 log bps</b>. The best DOWN forecast is the market-feature GRU16 policy, with MAE <b>232.3157 log bps</b>. Both are selected policies, not merely ungated diagnostic candidates. DOWN GRU beats its median baseline in all four 2024 quarters and January 2025; UP GRU does not outperform the original UP MLP.')+
        para('These are research results from repeatedly inspected historical evaluation data. They are not independent confirmation of future performance. MAE determines the rankings; the quarterly mean remains stronger on RMSE. Different experiments use different earlier-validation gates, so “selected” is not one universal acceptance standard.')))
    sections.append(section('Target and evaluation contract',para('UP and DOWN are separate nonnegative future excursion magnitudes over 72 hours, in log basis points. Conceptually, UP = max(0, maximum future log-price change) and DOWN = max(0, negative minimum future log-price change), over the canonical sampled price path. They are not signed endpoint returns or direction-conditioned targets. Valid zeros are retained.')+
        para(f'Every ranked forecast uses the same <b>18,385 origins per target</b>, from {esc(reference.decision_time.min())} through {esc(reference.decision_time.max())}. The feature-training inventory contains 48,000 eligible rows; 96 invalid development labels were excluded in the audited loaders. Quarterly issue dates are January, April, July and October 2024. The October fit is retained through January 2025. A training label is available only when origin + 72 hours is strictly earlier than the relevant boundary.')+
        para('MAE = mean absolute prediction error; RMSE = square root of mean squared error; Spearman IC = rank correlation between forecasts and realized magnitudes. Lower MAE and RMSE are better; higher IC indicates stronger ranking, not necessarily better sizing. Gain = 100 × (median MAE − model MAE) / median MAE. Overlapping 72-hour outcomes are dependent; row counts are not independent sample counts.')))
    for target in ['UP','DOWN']:
        view=ranking[ranking.target.eq(target)][['rank','experiment','forecast','MAE','RMSE','Spearman','gain_percent']]
        sections.append(section(target+' model ranking',table(view)+para('Ranked entries are complete saved forecast policies. “Candidate” ignores the final acceptance gate but was chosen on earlier validation. Individual architectures that were tested only inside validation cannot be ranked by outer MAE. Equal selected/candidate forecasts are shown once. The duplicated DOWN-only focused HGB run is represented by the identical both-target run. Incomplete runs and diagnostics without a new forecast are excluded.')))
    winners=[]
    for target,name in [('UP','Original neural MLP'),('DOWN','Market MLP and GRU search')]:
        frame=data[name];frame=frame[frame.target.str.startswith(target.lower())]
        for period,part in frame.groupby('period'):
            winners.append(dict(Target=target,Period=period,MAE=float(abs(part.actual-part.selected).mean()),
                Baseline_MAE=float(abs(part.actual-part.training_median).mean()),RMSE=float(np.sqrt(np.square(part.actual-part.selected).mean())),
                Spearman=float(part.actual.corr(part.selected,method='spearman'))))
    sections.append(section('Best model performance by period',table(winners)))
    neural=selections['Original neural MLP']
    up=[s for s in neural if s['target']=='up_magnitude_log_bps']
    up_cards=[]
    for s in up:
        spec=s['selected'];n=len(spec['features'])
        up_cards.append(dict(Issue=s['issue'][:10],Features=n,Architecture=f'{2*n} → 32 ReLU → 1 linear',Epochs=60,Model_weight=spec['weight']))
    sections.append(section('Best UP model architecture and hyperparameters',para('A separate regularized MLP is fitted at each quarterly issue. Each raw feature has an added missingness flag, doubling the input dimension. Training medians impute missing values; training means and standard deviations scale inputs, which are clipped to [−8, 8]. The target is divided by max(training median, 1). Linear output is rescaled and clipped at zero before blending with the quarterly median.')+
        table(up_cards)+para('All four selected UP models use AdamW, learning rate 0.001, weight decay 0.1, batch size 256, 60 epochs, L1 loss, gradient norm cap 1, seed 48, deterministic training and two CPU threads. All mature training rows are used. Search also tested hidden layers [32,16] for 40 epochs and [64,32] for 60 epochs, both with decay 0.01; packs were current features versus current plus long-price features. Median blend weights were 0, 0.25, 0.5, 0.75, 1. UP selected the current pack and [32] in every quarter.')+
        para('Original neural acceptance required at least 0.1% pooled MAE improvement over the median and wins in at least two of three earlier windows. It did not use the newer four-window maximum-loss gate.')))
    for s in up:
        sections.append(section('UP features for '+s['issue'][:7],para(f"Exact selected inventory: {len(s['selected']['features'])} raw inputs. Feature names identify the source family, measurement and lookback.")+table([{'Feature':c} for c in s['selected']['features']])))
    down=selections['Market MLP and GRU search']
    base_features=[c.split('__',1)[1] for c in down[0]['selected']['features'] if c.startswith('SEQ0__')]
    sections.append(section('Best DOWN GRU architecture and hyperparameters',para('The GRU sees four time-ordered snapshots at t−18h, t−12h, t−6h and t. Each snapshot contains the same 30 causal engineered features plus 30 missingness flags: tensor shape <b>[batch, 4, 60]</b>. The one-layer, unidirectional GRU has 16 hidden units, no dropout and a linear 16→1 head. The final hidden state produces the forecast adjustment; there is no teacher forcing or future sequence input.')+
        para('Let m be the mature training median, u = max(m,1), and a the final linear head output. The raw forecast is m + 0.75u tanh(a). The deployed output is max(0, m + w × 0.75u tanh(a)); w is set to zero when fewer than 80% of the sequence inputs are finite. Thus the reported GRU is already regularized toward the median, not an unrestricted raw sequence regression.')+
        table([{'Parameter':'Optimizer / learning rate','Value':'AdamW / 0.001'},{'Parameter':'Weight decay / loss','Value':'0.03 / MAE on constant-scaled target residual'},
               {'Parameter':'Epochs / batch size','Value':'24 / 512'},{'Parameter':'Estimator rows','Value':'Up to 4096 evenly spaced mature rows'},
               {'Parameter':'Preprocessing','Value':'Full mature training rows; median imputation, flags, standardization, clip ±8'},
               {'Parameter':'Initialization / gradient cap','Value':'Seed 48; head weights and bias zero; gradient norm 1'},
               {'Parameter':'GRU trainable parameters','Value':'3761 = 3744 recurrent parameters + 17 output-head parameters'},
               {'Parameter':'Quarterly model weights','Value':'January 0.5; April 0.5; July 0.5; October 0.25'}])+
        para('The search tested MLP16 (learning rate 0.002, decay 0.1), MLP32→16 (0.001, 0.03), GRU8 (0.002, 0.1) and GRU16 (0.001, 0.03), each for 24 epochs and blend weights 0.1, 0.25, 0.5, 1. This was a compact paired-configuration search, not a full Cartesian hyperparameter sweep. GRU16 won and passed the gate at every issue. The UP GRU follow-up tested only GRU8 and GRU16 and selected GRU8, but did not beat the earlier UP MLP.')+
        para('The newer gate requires at least 0.1% pooled MAE improvement over the median, wins or ties in at least three of four earlier purged quarters, and no window more than 2% worse. Final fitting uses only mature labels. The test period never selects the quarterly architecture or blend.')))
    sections.append(section('Exact DOWN GRU feature inventory',table([{'Feature at each snapshot':c} for c in base_features])+para('The 30 raw inputs comprise six price measures, twenty market summaries, and four interactions. Four snapshots produce 120 values; missingness flags double this to 240 scalar inputs organized as four 60-channel steps. No univariate correlation selection is applied to this pack. Each training input requires at least 80% finite coverage and 1,000 observed values; otherwise the neural pack falls back to its median.')))
    formulas=[
        ('NEWPRICE realized volatility','sqrt(sum r²), using strictly earlier 5-minute returns within the lookback'),
        ('NEWPRICE upside and downside volatility','sqrt(sum max(r,0)²) and sqrt(sum min(r,0)²)'),
        ('NEWPRICE net and range','sum r; local maximum minus minimum of cumulative log price on sampled 5-minute endpoints'),
        ('NEWPRICE distances and largest move','local high minus current level; current level minus local low; max |r|'),
        ('NEWPRICE volatility ratios','(RV_short / sqrt(short hours)) / (RV_long / sqrt(long hours)); denominator floor 1e−6'),
        ('LONG hourly construction','R = sum of exactly 12 finite 5-minute returns in [t−1h,t). Incomplete hours remain missing'),
        ('72h hourly volatility','sqrt(sum R² / 72)'),
        ('72h normalized momentum','sum R / max(sqrt(sum R²), 1e−8)'),
        ('72h range position','(current cumulative hourly level − local low) / max(local high − local low, 1e−8)'),
        ('72h upside variance share','sum max(R,0)² / max(sum R², 1e−16)'),
        ('168h drawdown in volatility units','(local high − current level) / max(sqrt(sum R²), 1e−8)'),
        ('336h jump variance share','max R² / max(sum R², 1e−16)'),
        ('Market rolling mean','Mean of observed half-hour snapshots in [t−h,t), with at least ceil(0.8 × 2h) observations'),
        ('Market deviation','Snapshot at t−30min minus the same historical rolling mean'),
        ('Liquidation volatility interaction','Mean historical liquidation activity rate × hourly price volatility, same h-hour window'),
        ('Flow momentum interaction','(mean flow-pressure percentile − 0.5) × normalized momentum, same h-hour window'),
        ('GRU lag sequence','Reindex engineered features at exact t−18h, t−12h, t−6h and t; absent snapshots remain missing')]
    sections.append(section('How the features were created',para('Features are mathematical measurements defined before model evaluation. They are computed from canonical return history and existing masked market-source snapshots; the model learns how to combine them. A strong score is not evidence that an interaction is causal.')+table(formulas,['Feature family','Formula or construction'])+
        para('For the winning DOWN pack, the market inputs are flow pressure, flow activity, local historical liquidation activity rate, peer historical liquidation activity rate, and liquidation venue breadth. Each supplies mean and deviation at 24h and 72h: 5 × 2 × 2 = 20 features. Liquidation×volatility and flow×momentum at those two lookbacks add four interactions. Flow activity is not signed flow; historical liquidation activity rate is not current liquidation size.')+
        para('The broader fresh-price universe contained 104 LONG features: 12 measurements at 24, 36, 72, 144, 168, 336, 720 and 1440 hours, plus four volatility ratios and four drift differences. Sixteen FRESH72 features added normalized P90 and negative P10 tails, adjacent-return products divided by mean squared return, and dispersion of absolute returns at 24,72,168,720h. The broader market universe added 187 snapshot/rolling summaries and 18 interactions, giving 325 total candidates with the 120 fresh price inputs. The focused packs reduced this to 30 flow/liquidation, 38 leverage, or 62 combined inputs.')+
        para('Legacy F4Z features in the winning UP model are inherited masked percentile, robust-score and activity-rate inputs. This report verifies their exact selected names and use, but does not independently reconstruct the upstream F4Z normalization algorithms. Do not interpret their names as newly audited formulas. Full generated feature definitions and market source mappings are included below.')))
    families=[
      ('Original and fresh tree search','Ridge alpha = 0.01 × training rows; or MAE HGB: 80 iterations, learning rate .05, depth 3, 7 leaves, leaf minimum 128, L2=1, seed48, no early stopping. Depth2 MAE decision tree with leaf128 used for screening.'),
      ('Tuned original trees','Same original HGB architecture; searches current versus extended price inputs, all/180/365-day mature training history, and model/median weights0/.25/.5/.75/1.'),
      ('Compact trees','MAE HGB; depths1/2/3, leaves2/4/8, minimum leaves256/256/512;80 iterations, rate.05,L2=5;blend.25/.5/.75/1.'),
      ('Transformed spline and ExtraTrees','Raw median/IQR, signed log, or empirical midrank transforms. Spline:4 quantile knots,degree2,constant extrapolation + median QuantileRegressor alpha.01. ExtraTrees:64 trees,absolute error,depth5,leaf64,max_features1,bootstrap cap4096,seed48.'),
      ('Guarded residual and sizing','Residual linear/tree, quantile mapping, 3/5-bin classifiers. Shared MAE HGB residual tree:depth2,4leaves,leaf256,80iterations,rate.05,L2=10. Stable-sign/coverage screening; deadzones0/.05/.1 and blends.25/.5/1.'),
      ('Guarded linear and classification details','SGDRegressor:epsilon-insensitive loss with epsilon0,elastic-net alpha.01,l1_ratio.5,max_iter3000,tol1e-5,invscaling eta0.001,power_t.25,no shuffle,seed48. Histogram classifier:depth2,4leaves,leaf256,80iterations,rate.05,L2=10;3/5 quantile-defined classes mapped to historical class medians. Quantile mapping holds out a purged90-day calibration slice.'),
      ('Quarterly anchor','Same residual tree; compares quarterly versus rolling anchors, up to6 stable/redundancy-filtered features, blends.1/.25/.5; missing inputs revert to anchor.'),
      ('Fast market HGB','MAE HGB:depth3,7leaves,leaf64,60iterations,rate.05,L2=10,max_bins63,6000row cap. Price36 or market-plus-price241;blends.5/1.'),
      ('Focused HGB','MAE HGB:depth3,7leaves,leaf128,60iterations,rate.05,L2=10,max_bins63,6000row cap. Packs38/30/62;blends.1/.25/.5/1. DOWN candidate uses30feature flow pack all quarters.'),
      ('Best-model calibration','No underlying model refit. Rolling matured residual medians30/60/120days,minimum240observations,weights.1/.25/.5;offset capped25% of quarterly median;earlier90day validation. Worsened both previous best models.'),
      ('Rolling median policy','No neural or tree architecture: causal expanding and rolling30/90/180/365-day mature-label medians;minimum48observations, chosen on earlier data. Windows are measured over label maturity timestamps.')]
    sections.append(section('Other model architectures and hyperparameters',table(families,['Experiment family','Verified configuration summary'])+para('A ranked experiment can switch model family or fall back to the median by issue. Exact saved selection plans and contract settings below identify what was actually used; search-space settings do not imply every model was deployed. Source files are retained in the original run folders.')))
    sections.append(section('Feature selection and leakage controls',para('The early searches screened candidate features with shallow trees or simple models, removed redundant features, and tested additions on earlier windows. Compact search used training coverage and variation checks, single-feature screening, Spearman redundancy filtering and bounded greedy selection. Transformation experiments tested small fixed packs. Guarded experiments screened quarterly sign stability and missingness. The focused HGB and neural sequence experiments instead admitted fixed economic feature groups jointly, allowing interactions that a univariate screen might miss.')+
        para('All imputations, scales, model parameters and label medians must be learned on mature training rows. Rolling market windows exclude the forecast timestamp; complete hourly price windows use only earlier endpoints. No future filling is used. The neural estimator cap subsamples only training rows; validation and evaluation retain all origins. For GRU, historical snapshots are assembled before splitting but reference only earlier feature timestamps. Prefix/future-perturbation and synthetic replay checks were performed during runner preparation.')+
        para('The older MLP uses three earlier validation windows; the newer HGB/GRU uses four. Calibration and rolling-label policies update from newly matured labels during evaluation, so their information schedules differ from a frozen quarterly neural model. These distinctions should be retained when interpreting the pooled ranking.')))
    sections.append(section('Evidence and limitations',para('This report re-read completed runs, verified prediction-file SHA256 hashes, aligned origin keys and actual targets, and recomputed pooled MAE, RMSE and Spearman from saved predictions. It did not train models or rerun model inference. Replay success is a saved-run claim, distinct from the report’s independent prediction and metric checks. No confidence interval or significance test is asserted. Repeated experimentation on this same evaluation period creates selection bias; an untouched later period is required before claiming a durable improvement.')+
        para('The DOWN GRU improves MAE across every reported period against the median, but does not beat the quarterly mean on pooled RMSE. The best UP architecture is not the best DOWN architecture. Do not replace the stronger UP MLP with its weaker GRU follow-up solely for architectural uniformity.')))
    appendix=''
    for name,relative in RUNS:
        appendix+=f'<h3>{esc(name)}</h3><p><a href="../../Code/experiments/{esc(relative)}/SUMMARY.txt">Saved summary</a> · <a href="../../Code/experiments/{esc(relative)}/predictions.csv">Predictions</a></p>'
        appendix+='<details><summary>Saved implementation sources</summary><ul>'+''.join(f'<li><a href="../../Code/experiments/{esc(relative)}/{esc(p.name)}">{esc(p.name)}</a></li>' for p in sorted((EXP/relative).glob('*.py')))+'</ul></details>'
        contract=contracts[name]
        settings={k:v for k,v in contract.items() if k not in ['input_hashes','source_hashes','guidance_hashes','candidate_features']}
        appendix+=detail('Exact contract settings',settings)+detail('Exact quarterly selections and features',selections[name])
    featurepath=EXP/RUNS[12][1]/'feature_dictionary.csv'
    definitions=pd.read_csv(featurepath).fillna('')
    appendix+='<h3>Generated feature dictionary</h3><details><summary>All generated price and market formulas</summary>'+table(definitions)+'</details>'
    appendix+=detail('Prediction provenance and audit',audit)
    sections.append(section('Source linked technical appendix',appendix))
    css='''body{font-family:Segoe UI,Arial,sans-serif;color:#17232d;max-width:1150px;margin:36px auto;padding:0 24px;line-height:1.5}h1{font-size:32px;color:#111}h2{font-size:24px;margin-top:40px;border-bottom:1px solid #cbd5df;padding-bottom:8px}h3{margin-top:28px}p{max-width:1050px}table{width:100%;border-collapse:collapse;margin:18px 0 24px;font-size:13px}th,td{border:1px solid #d9d9d9;padding:9px 10px;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#17354b;color:white}tr:nth-child(even){background:#f2f6f9}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f9;padding:15px;font-size:12px}summary{cursor:pointer;padding:9px;background:#eef3f7}details{margin:8px 0}a{color:#12618a}.meta{color:#52606c}@media print{body{margin:0;max-width:none;font-size:10pt}h2,h3{break-after:avoid}tr{break-inside:avoid}thead{display:table-header-group}table{font-size:8pt}a{color:inherit}details:not([open]){display:none}}'''
    document='<!doctype html><html lang="en"><head><meta charset="utf-8"><title>72 hour model research report</title><style>'+css+'</style></head><body><h1>72 hour model research report</h1><p class="meta">BTC canonical UP and DOWN excursions · Prepared 14 November 2026 · Saved experiment review</p>'+''.join(sections)+'</body></html>'
    (OUT/'report_72h.html').write_text(document,encoding='utf-8')
    # Deterministic report checks, not additional model tests.
    assert len(base_features)==30
    assert [len(s['selected']['features']) for s in up]==[21,19,51,2]
    assert ranking[ranking.target.eq('UP')].iloc[0].experiment=='Original neural MLP'
    assert ranking[ranking.target.eq('DOWN')].iloc[0].experiment=='Market MLP and GRU search'
    assert '<table' in document and document.count('<table')==document.count('</table>')
    print('Report:',OUT/'report_72h.html')
    print('Verified runs:',len(audit),'Ranked forecasts:',len(ranking),'Generated feature definitions:',len(definitions))
    print(ranking.groupby('target').head(3)[['target','experiment','forecast','MAE','RMSE','Spearman']].to_string(index=False))


if __name__=='__main__':main()
