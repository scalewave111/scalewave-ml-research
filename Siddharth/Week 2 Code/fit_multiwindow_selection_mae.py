"""Controlled one-window versus three-window validation selection. NEW CPU TRAINING.
Save beside fit_spline_median_mae.py in Week 2/Code.
Same candidates: linear_control, spline_light, spline_regularized. No model blending.
Three expanding inner fits at outer issue minus 9, 6, 3 months; evaluate first two
months following each inner cutoff. Final month is a gap. All labels must mature
before the respective three-month block ends. Original last window is preserved.
Choose by row-weighted validation MAE, then original candidate order on exact ties.
Both selection methods use the SAME outer fits and SAME test rows.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import fit_spline_median_mae as base

OFFSETS=[9,6,3]


def choose(records):
    return min(records,key=lambda r:(r['mae_bps'],list(base.CONFIGS).index(r['model'])))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--out',type=Path)
    args=ap.parse_args();ws=args.workspace.resolve();w1=ws.parent/'Week 1'
    out=args.out or ws/'Code/experiments/multiwindow_selection_mae_v1'
    base.check(not out.exists(),f'Preserving {out}; use --out with a new folder')
    previous=ws/'Code/experiments/spline_median_mae_v1'
    base.check((previous/'COMPLETE.json').exists(),'Completed spline_median_mae_v1 is required')
    base.check(json.loads((previous/'COMPLETE.json').read_text())['status']=='COMPLETED','Incomplete source spline run')
    source_path=Path(base.__file__).resolve()
    contract=json.loads((previous/'CONTRACT.json').read_text())
    base.check(hashlib.sha256(source_path.read_bytes()).hexdigest()==contract['source_sha256'],'Spline runner changed since previous experiment')
    x,y,dev,ref=joblib.load(w1/'Code/experiments/eight_feature_exhaustive_v1/prepared.joblib')
    base.check(x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing,'Invalid input index')
    base.check(dev.equals(ref.index) and len(dev)==18481,'Unexpected test population')
    base.check((y.label_end==y.index+pd.Timedelta(hours=48)).all(),'Unexpected target horizon')
    x=x[base.COLS].replace([np.inf,-np.inf],np.nan)
    signed=y.signed_return_log_bps.astype(float)
    target=pd.Series(np.abs(10000*np.expm1(signed.to_numpy()/10000)),index=y.index)
    side=pd.Series(np.where(signed>0,'UP',np.where(signed<0,'DOWN','FLAT')),index=y.index)
    old=pd.read_csv(w1/'Code/experiments/conditional_48h_magnitude_calibration_diagnostics_v1/predictions.csv',index_col='decision_time')
    old.index=pd.to_datetime(old.index,utc=True);old['issue']=pd.to_datetime(old.issue,utc=True)
    base.check(old.index.equals(dev.sort_values()),'Reference index mismatch')
    base.check(np.array_equal(old.actual_direction,side.loc[old.index]),'Reference direction mismatch')
    np.testing.assert_allclose(old.actual_magnitude_bps,target.loc[old.index],rtol=0,atol=1e-8)
    base.check(set(old.issue.unique())==set(base.STARTS),'Unexpected outer folds')
    prior_val=pd.read_csv(previous/'validation_scores.csv');prior_val['issue']=pd.to_datetime(prior_val.issue,utc=True)
    prior_pred=pd.read_csv(previous/'predictions.csv',index_col='decision_time');prior_pred.index=pd.to_datetime(prior_pred.index,utc=True)
    base.check(prior_pred.index.equals(old.index),'Prior spline prediction index mismatch')
    # Preflight all folds before beginning training; never silently shorten the protocol.
    splits={}
    for issue in base.STARTS:
        for direction in ['UP','DOWN']:
            for months in OFFSETS:
                cutoff=issue-pd.DateOffset(months=months)
                block_end=cutoff+pd.DateOffset(months=3)
                train=y.index[(y.label_end<cutoff)&side.eq(direction)]
                val=y.index[(y.index>=cutoff)&(y.index<cutoff+pd.DateOffset(months=2))&(y.label_end<block_end)&side.eq(direction)]
                base.check(len(train)>=512 and len(val)>=100,f'Insufficient history for {issue.date()} {direction} offset {months}; protocol not changed')
                base.check(y.loc[train,'label_end'].max()<cutoff and y.loc[val,'label_end'].max()<block_end and block_end<=issue,'Purging violation')
                splits[(issue,direction,months)]=(cutoff,train,val)
    out.mkdir(parents=True)
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (out/'spline_recipe_snapshot.py').write_bytes(source_path.read_bytes())
    base.dump(out/'CONTRACT.json',dict(candidates=base.CONFIGS,offset_months=OFFSETS,
        selection='Single: latest two-month window. Multi: total absolute error / total validation rows across three windows. Exact MAE ties: original config order.',
        tie_note='Both methods use same MAE-only tie rule; differs from previous runner only on exact MAE ties (which previously used IC).',
        models='One standalone selected model; no ensemble or neural addition',
        outer='Same saved quarterly outer fits from spline experiment, replay verified; only selection differs',
        limitation='Historically exposed data; this protocol was proposed after inspecting past outcomes. No independent confirmation or significance claims.'))
    frames=[];validation=[];selections=[];audits=[]
    for issue in base.STARTS:
        te=old.index[old.issue.eq(issue)]
        frame=pd.DataFrame(dict(issue=str(issue),actual_direction=side.loc[te],actual_magnitude_bps=target.loc[te]),index=te)
        for direction in ['UP','DOWN']:
            print(f'START {issue.date()} {direction}',flush=True)
            by_model={name:[] for name in base.CONFIGS}
            for months in OFFSETS:
                cutoff,train,val=splits[(issue,direction,months)]
                for name,config in base.CONFIGS.items():
                    print(f'  Validation from {cutoff.date()}: {name}',flush=True)
                    if months==3:
                        # Reuse original inner fits for the latest window, rather than refit them.
                        saved=joblib.load(previous/f'{issue:%Y-%m}_{direction}_{name}.joblib')
                        base.check(saved['inner_train_keys'].equals(train) and saved['validation_keys'].equals(val),'Prior inner split mismatch')
                        art=saved['inner']
                    else: art=base.fit_model(x.loc[train],target.loc[train],config)
                    vp=base.predict(art,x.loc[val]);m=base.metrics(target.loc[val],vp)
                    rec=dict(issue=str(issue),side=direction,offset_months=months,validation_start=str(cutoff),model=name,**m)
                    validation.append(rec);by_model[name].append(rec)
                    if months==3:
                        match=prior_val[(prior_val.issue==issue)&(prior_val.side==direction)&(prior_val.model==name)]
                        base.check(len(match)==1,'Missing prior validation score')
                        np.testing.assert_allclose(m['mae_bps'],match.iloc[0].mae_bps,rtol=0,atol=1e-8)
                    else:
                        joblib.dump(dict(artifact=art,train_keys=train,validation_keys=val,cutoff=cutoff),out/f'{issue:%Y-%m}_{direction}_{name}_inner{months}.joblib',compress=3)
            single=choose([r[-1] for r in by_model.values()])
            totals=[dict(model=name,mae_bps=float(sum(r['mae_bps']*r['rows'] for r in rs)/sum(r['rows'] for r in rs))) for name,rs in by_model.items()]
            multi=choose(totals)
            # Same final fitted candidates for both selection methods; zero extra outer tuning.
            train_outer=y.index[(y.label_end<issue)&side.eq(direction)]
            for name in base.CONFIGS:
                saved=joblib.load(previous/f'{issue:%Y-%m}_{direction}_{name}.joblib')
                base.check(saved['train_keys'].equals(train_outer),'Outer train keys differ')
                p=base.predict(saved['outer'],x.loc[te])
                np.testing.assert_allclose(p,prior_pred.loc[te,f'{direction}_{name}'],rtol=0,atol=1e-8)
                frame[f'{direction}_{name}']=p
                audits.append(dict(issue=str(issue),side=direction,model=name,max_outer_replay_error_bps=float(np.max(np.abs(p-prior_pred.loc[te,f'{direction}_{name}'])))))
            frame[f'{direction}_single_window']=frame[f'{direction}_{single["model"]}']
            frame[f'{direction}_three_windows']=frame[f'{direction}_{multi["model"]}']
            selections.append(dict(issue=str(issue),side=direction,single_selected=single['model'],multi_selected=multi['model'],multi_validation_scores=totals))
            for name in ['mlp','hybrid']:
                frame[f'{direction}_existing_{name}']=old.loc[te,f'{name}_{direction.lower()}_magnitude_bps']
            for b in ['mean','median']:
                frame[f'{direction}_baseline_{b}']=old.loc[te,f'{direction.lower()}_train_{b}_baseline_bps']
                np.testing.assert_allclose(frame[f'{direction}_baseline_{b}'],getattr(target.loc[train_outer],b)(),rtol=0,atol=1e-8)
            print(f'DONE {direction}: single={single["model"]}, three={multi["model"]}',flush=True)
        frames.append(frame);frame.to_csv(out/f'{issue:%Y-%m}_predictions.csv',index_label='decision_time')
        pd.DataFrame(validation).to_csv(out/'validation_scores.csv',index=False)
        base.dump(out/'SELECTIONS.json',selections);base.dump(out/'REPLAY_AUDIT.json',audits)
    pred=pd.concat(frames).sort_index();base.check(pred.index.equals(old.index),'Final index mismatch')
    rows=[]
    for issue,f in [('POOLED',pred)]+list(pred.groupby('issue')):
        for direction in ['UP','DOWN']:
            g=f[f.actual_direction.eq(direction)];actual=g.actual_magnitude_bps
            reference='existing_mlp' if direction=='UP' else 'existing_hybrid'
            rm=base.metrics(actual,g[f'{direction}_{reference}'])['mae_bps']
            sm=base.metrics(actual,g[f'{direction}_single_window'])['mae_bps']
            for col in [c for c in pred if c.startswith(direction+'_')]:
                m=base.metrics(actual,g[col]);rows.append(dict(issue=issue,side=direction,model=col[len(direction)+1:],**m,
                    mae_gain_vs_single_window=sm-m['mae_bps'],current_reference=reference,mae_gain_vs_current=rm-m['mae_bps']))
    scores=pd.DataFrame(rows);scores.to_csv(out/'scores.csv',index=False);pred.to_csv(out/'predictions.csv',index_label='decision_time')
    summary=('ONE VERSUS THREE VALIDATION WINDOWS\n'
        'Same candidates and final fitted models. Only validation evidence used to choose the model changes.\n'
        'Multi-window score is row-weighted MAE across earlier windows. IC is reported, not optimized.\n'
        'Positive MAE gain means improvement; historical exposed results remain exploratory.\n'
        'All training and validation outcomes mature before their allowed cutoffs. No test-based selection.\n'
        'Pooled baseline IC is not within-fold ranking. No significance tests.\n\n'
        +scores[scores.issue.eq('POOLED')].to_string(index=False)+'\n\nSELECTIONS\n'
        +pd.DataFrame(selections).drop(columns='multi_validation_scores').to_string(index=False))
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    base.dump(out/'COMPLETE.json',dict(status='COMPLETED',rows=len(pred),outer_replay_checks=len(audits)))
    print(summary);print(f'Outputs: {out}')


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
