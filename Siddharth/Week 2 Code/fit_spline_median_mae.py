"""Small additive spline median regression; NEW CPU TRAINING on execution.
Save in Week 2/Code. Standalone model, no neural network or model blending.
Uses the same eight features, expanding quarterly folds and ordinary-bps targets.
Requires numpy pandas scipy scikit-learn joblib threadpoolctl. No GPU or torch.
"""
from pathlib import Path
import argparse
import json
import hashlib
import importlib.metadata
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import QuantileRegressor
from threadpoolctl import threadpool_limits

COLS=['F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__15m__value',
'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__5m__value',
'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value',
'PRICE__vol_ratio_4h_24h','PRICE__realized_vol_168h',
'F4Z__liquidation__venue_breadth_percentile__1h__value',
'F4Z__funding__funding_persistence_percentile__5m__value',
'F4Z__basis__basis_level_rz__1h__value']
STARTS=pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'],utc=True)
CONFIGS={'linear_control':{'spline':False,'alpha':.0001},
         'spline_light':{'spline':True,'alpha':.0001},
         'spline_regularized':{'spline':True,'alpha':.001}}


def check(ok,msg):
    if not ok: raise ValueError(msg)


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf-8')


def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    check(len(y)>0 and y.shape==p.shape and np.isfinite(y).all() and np.isfinite(p).all(),'Invalid score arrays')
    ic=pd.Series(y).corr(pd.Series(p),method='spearman') if np.ptp(y)>0 and np.ptp(p)>0 else np.nan
    return dict(rows=len(y),mae_bps=float(np.abs(y-p).mean()),rank_ic=float(ic) if np.isfinite(ic) else None)


def prep(x):
    med=x.median().fillna(0.);f=x.fillna(med)
    scale=f.quantile(.75)-f.quantile(.25)
    scale=scale.where(scale>1e-6,f.std(ddof=0)).fillna(1.).clip(lower=1e-6)
    return dict(median=med,scale=scale)


def design(x,p):
    z=((x.fillna(p['median'])-p['median'])/p['scale']).clip(-8,8).to_numpy(np.float32).astype(float)
    check(np.isfinite(z).all(),'Nonfinite design')
    return z


def fit_model(x,y,config):
    p=prep(x);z=design(x,p);transformer=None
    active=np.flatnonzero(np.ptp(z,axis=0)>1e-10)
    if config['spline']:
        # Uniform knots avoid duplicate quantile knots for saturated/tied features.
        # Constants are excluded using TRAINING values only; the intercept absorbs them.
        if len(active):
            transformer=SplineTransformer(n_knots=3,degree=2,knots='uniform',
                                          extrapolation='constant',include_bias=False)
            features=transformer.fit_transform(z[:,active])
        else: features=np.zeros((len(z),1))
    else: features=z
    model=QuantileRegressor(quantile=.5,alpha=config['alpha'],solver='highs')
    # Match original median component's target scaling/precision.
    target=np.asarray(y,np.float32)/100.
    model.fit(features,target.astype(float))
    return dict(prep=p,active=active,transformer=transformer,model=model,config=config,
                design_columns=features.shape[1])


def predict(art,x):
    z=design(x,art['prep'])
    if art['config']['spline']:
        features=art['transformer'].transform(z[:,art['active']]) if art['transformer'] is not None else np.zeros((len(z),1))
    else: features=z
    return np.maximum(art['model'].predict(features)*100.,0.)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--out',type=Path)
    args=ap.parse_args();ws=args.workspace.resolve();w1=ws.parent/'Week 1'
    out=args.out or ws/'Code/experiments/spline_median_mae_v1'
    check(not out.exists(),f'Preserving output {out}; use --out with a new folder')
    x,y,dev,ref=joblib.load(w1/'Code/experiments/eight_feature_exhaustive_v1/prepared.joblib')
    check(x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing,'Invalid dataset index')
    check(dev.equals(ref.index) and len(dev)==18481,'Unexpected evaluation population')
    check((y.label_end==y.index+pd.Timedelta(hours=48)).all(),'Unexpected outcome horizon')
    x=x[COLS].replace([np.inf,-np.inf],np.nan)
    signed=y.signed_return_log_bps.astype(float)
    target=pd.Series(np.abs(10000*np.expm1(signed.to_numpy()/10000)),index=y.index)
    side=pd.Series(np.where(signed>0,'UP',np.where(signed<0,'DOWN','FLAT')),index=y.index)
    old=pd.read_csv(w1/'Code/experiments/conditional_48h_magnitude_calibration_diagnostics_v1/predictions.csv',index_col='decision_time')
    old.index=pd.to_datetime(old.index,utc=True);old['issue']=pd.to_datetime(old.issue,utc=True)
    check(old.index.equals(dev.sort_values()),'Reference timestamps differ')
    np.testing.assert_allclose(old.actual_magnitude_bps,target.loc[old.index],rtol=0,atol=1e-8)
    check(np.array_equal(old.actual_direction,side.loc[old.index]),'Reference directions differ')
    check(old.actual_direction.value_counts().to_dict()=={'UP':10191,'DOWN':8290},'Unexpected side counts')
    check(set(old.issue.unique())==set(STARTS),'Unexpected folds')
    out.mkdir(parents=True)
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'CONTRACT.json',dict(configs=CONFIGS,splines='3 uniform knots, degree 2, additive only, constant extrapolation; no interactions',
        preprocessing='Train-only median/IQR with std fallback, clip [-8,8], float32 then float64; training constants omitted from splines only',
        objective='Median pinball loss; target ordinary bps divided by 100; nonnegative predictions',
        selection='Earlier first two months of prior quarter; validation MAE then IC then declared config order; linear control is eligible',
        history='Expanding; all training label_end strictly before cutoff',
        evaluation='Matching realized direction, original test rows unchanged; historical exposed results exploratory',
        linear_control='Same original MLP median-regression recipe, evaluated in float64; not a reproduction of the full MLP or RF component',
        versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','joblib']},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    frames=[];validation=[];selections=[];audits=[]
    for issue in STARTS:
        inner=issue-pd.DateOffset(months=3)
        vk=y.index[(y.index>=inner)&(y.index<issue-pd.DateOffset(months=1))&(y.label_end<issue)]
        te=old.index[old.issue.eq(issue)]
        frame=pd.DataFrame(dict(issue=str(issue),actual_direction=side.loc[te],actual_magnitude_bps=target.loc[te]),index=te)
        for direction in ['UP','DOWN']:
            ti=y.index[(y.label_end<inner)&side.eq(direction)]
            tr=y.index[(y.label_end<issue)&side.eq(direction)]
            va=vk[side.loc[vk].eq(direction)]
            check(min(len(ti),len(tr),len(va))>=100,'Too few fold rows')
            check(y.loc[ti,'label_end'].max()<inner and y.loc[tr,'label_end'].max()<issue,'Purge failed')
            print(f'START {issue.date()} {direction}: {len(ti)} inner / {len(tr)} outer rows',flush=True)
            candidates=[]
            for name,config in CONFIGS.items():
                print(f'  Fitting {name}',flush=True)
                inn=fit_model(x.loc[ti],target.loc[ti],config)
                v=predict(inn,x.loc[va]);rec=dict(issue=str(issue),side=direction,model=name,**metrics(target.loc[va],v))
                candidates.append(rec);validation.append(rec)
                outer=fit_model(x.loc[tr],target.loc[tr],config);p=predict(outer,x.loc[te])
                frame[f'{direction}_{name}']=p
                path=out/f'{issue:%Y-%m}_{direction}_{name}.joblib'
                joblib.dump(dict(inner=inn,outer=outer,train_keys=tr,inner_train_keys=ti,validation_keys=va,issue=issue),path,compress=3)
                replay=predict(joblib.load(path)['outer'],x.loc[te])
                np.testing.assert_allclose(replay,p,rtol=0,atol=1e-8)
                audits.append(dict(issue=str(issue),side=direction,model=name,design_columns=outer['design_columns'],
                    active_feature_columns=[COLS[i] for i in outer['active']],max_replay_error_bps=float(np.max(np.abs(replay-p)))))
                print(f'  {name}: validation MAE {rec["mae_bps"]:.3f}, IC {rec["rank_ic"]}',flush=True)
            win=min(candidates,key=lambda r:(r['mae_bps'],-r['rank_ic'] if r['rank_ic'] is not None else float('inf'),list(CONFIGS).index(r['model'])))
            selections.append(win);frame[f'{direction}_validation_selected']=frame[f'{direction}_{win["model"]}']
            for name in ['mlp','hybrid']:
                frame[f'{direction}_existing_{name}']=old.loc[te,f'{name}_{direction.lower()}_magnitude_bps']
            for base in ['mean','median']:
                value=float(getattr(target.loc[tr],base)())
                frame[f'{direction}_baseline_{base}']=value
                np.testing.assert_allclose(old.loc[te,f'{direction.lower()}_train_{base}_baseline_bps'],value,rtol=0,atol=1e-8)
            print(f'DONE {direction}: validation selected {win["model"]}',flush=True)
        frame.to_csv(out/f'{issue:%Y-%m}_predictions.csv',index_label='decision_time');frames.append(frame)
        pd.DataFrame(validation).to_csv(out/'validation_scores.csv',index=False)
        dump(out/'SELECTIONS.json',selections);dump(out/'FIT_AUDIT.json',audits)
    pred=pd.concat(frames).sort_index();check(pred.index.equals(old.index),'Final timestamps differ')
    rows=[]
    for issue,f in [('POOLED',pred)]+list(pred.groupby('issue')):
        for direction in ['UP','DOWN']:
            g=f[f.actual_direction.eq(direction)];actual=g.actual_magnitude_bps
            reference='existing_mlp' if direction=='UP' else 'existing_hybrid'
            refmae=metrics(actual,g[f'{direction}_{reference}'])['mae_bps']
            linear=metrics(actual,g[f'{direction}_linear_control'])['mae_bps']
            bases={b:metrics(actual,g[f'{direction}_baseline_{b}'])['mae_bps'] for b in ['mean','median']}
            for col in [c for c in pred if c.startswith(direction+'_')]:
                m=metrics(actual,g[col]);rows.append(dict(issue=issue,side=direction,model=col[len(direction)+1:],**m,
                    mae_gain_vs_linear=linear-m['mae_bps'],current_reference=reference,mae_gain_vs_current=refmae-m['mae_bps'],
                    mae_gain_vs_mean=bases['mean']-m['mae_bps'],mae_gain_vs_median=bases['median']-m['mae_bps']))
    scores=pd.DataFrame(rows);scores.to_csv(out/'scores.csv',index=False)
    pred.to_csv(out/'predictions.csv',index_label='decision_time')
    summary=('ADDITIVE SPLINE MEDIAN REGRESSION\n'
        'Standalone median models. Same eight inputs, expanding quarterly training and test rows.\n'
        'Positive MAE gain means improvement. Rank IC is Spearman correlation, not accuracy percent.\n'
        'Validation selection includes the linear control; no test selection inside runner.\n'
        'Spline knots and preprocessing use training data only; no feature interactions or neural network.\n'
        'Baseline within-fold IC is undefined. Historical exposed comparison; no significance claims.\n\n'
        +scores[scores.issue.eq('POOLED')].to_string(index=False))
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    dump(out/'COMPLETE.json',dict(status='COMPLETED',rows=len(pred),replay_checks=len(audits)))
    print(summary);print(f'Outputs: {out}')


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
