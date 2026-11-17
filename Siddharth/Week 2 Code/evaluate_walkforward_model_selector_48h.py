r"""Compare monthly saved-model selection with fixed models and hindsight oracle.
python .\Code\evaluate_walkforward_model_selector_48h.py
No training. Strict 48-hour outcome maturity. Historical diagnostic only.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import tune_conditional_mean_magnitudes_48h as b

FAMILIES=['decision_tree','random_forest','extra_trees','gradient_boosting',
          'hist_gradient_boosting','xgboost','lightgbm','catboost']
WINDOWS=[30,60,90]


def load_run(folder):
    done=json.loads((folder/'COMPLETE.json').read_text())
    path=folder/'signed_return_predictions.csv'
    b.check(done['status']=='COMPLETE' and b.sha(path)==done['prediction_sha256'],'Prediction integrity failed')
    b.check(b.sha(folder/'CONTRACT.json')==done['contract_sha256'],'Contract integrity failed')
    f=pd.read_csv(path,index_col='decision_time');f.index=pd.to_datetime(f.index,utc=True).as_unit('ns')
    f['issue']=pd.to_datetime(f.issue,utc=True)
    b.check(f.index.is_unique and not f.index.hasnans,'Invalid timestamps')
    b.check(f.symbol.eq('BTCUSDT').all() and f.horizon_hours.eq(48).all(),'Wrong symbol/horizon')
    b.check((f.issue<=f.index).all(),'Model issue is after forecast')
    audit=json.loads((folder/'FIT_AUDIT.json').read_text())
    b.check(all(pd.Timestamp(a['max_training_label_end'])<pd.Timestamp(a['issue']) for a in audit),'Training label maturity failed')
    return f.sort_index()


def monthly_select(predictions,actual,days,min_rows):
    """Select a FAMILY STREAM once per month; only previously matured outcomes."""
    index=predictions.index
    month_keys=index.strftime('%Y-%m')
    output=pd.Series(np.nan,index=index,dtype=float)
    chosen=pd.Series('',index=index,dtype=object)
    ledger=[];scores=[]
    for month in sorted(set(month_keys)):
        cutoff=pd.Timestamp(month+'-01',tz='UTC')
        start=cutoff-pd.Timedelta(days=days)
        eligible=(index>=start)&(index+pd.Timedelta(hours=48)<cutoff)
        keys=index[eligible]
        enough=(start>=index.min() and len(keys)>=min_rows)
        winner='ridge';reason='insufficient_history_ridge_fallback';mse=None
        if enough:
            values=((predictions.loc[keys].sub(actual.loc[keys],axis=0))**2).mean()
            winner=values.idxmin();mse=float(values[winner]);reason='lowest_matured_window_mse'
            for model,value in values.items():
                scores.append(dict(window_days=days,selection_time=cutoff,model=model,rows=len(keys),mse_bps2=float(value)))
        take=index[month_keys==month]
        output.loc[take]=predictions.loc[take,winner]
        chosen.loc[take]=winner
        latest=keys.max()+pd.Timedelta(hours=48) if len(keys) else None
        b.check(latest is None or latest<cutoff,'Selector used incomplete future outcome')
        ledger.append(dict(window_days=days,selection_time=cutoff,window_start=start,
            eligible_rows=len(keys),max_used_label_end=latest if enough else None,
            selected_model=winner,reason=reason,selection_mse=mse,forecast_rows=len(take)))
    return output,chosen,ledger,scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tree-run',type=Path,default=b.EXP/'tree_signed_return_search_48h_v1/20261112T113518221929Z')
    p.add_argument('--interaction-run',type=Path,default=b.EXP/'tree_interaction_signed_search_48h_v1/20261112T125303170547Z')
    p.add_argument('--freeze-date',default='2024-07-01',help='Choose one window using only outcomes matured before this UTC month-start')
    p.add_argument('--min-rows',type=int,default=500)
    args=p.parse_args();freeze=pd.Timestamp(args.freeze_date)
    freeze=freeze.tz_localize('UTC') if freeze.tzinfo is None else freeze.tz_convert('UTC')
    b.check(freeze.day==1 and freeze==freeze.normalize() and args.min_rows>0,'Freeze date must be UTC month start; min rows positive')
    original=load_run(args.tree_run);interaction=load_run(args.interaction_run)
    b.check(original.index.equals(interaction.index),'Runs do not have identical forecast timestamps')
    b.check(original.issue.equals(interaction.issue),'Quarterly refit sections differ')
    for col in ['actual_signed_return_bps','p_up','p_down','ridge_signed_bps','side_means_signed_bps']:
        np.testing.assert_allclose(original[col],interaction[col],rtol=0,atol=1e-7)
    index=original.index;actual=original.actual_signed_return_bps
    b.check(index.min()<freeze<index.max(),'Freeze date must lie within saved forecast coverage')
    candidates=pd.DataFrame({'ridge':original.ridge_signed_bps},index=index)
    for label,f in [('tree',original),('interaction',interaction)]:
        for family in FAMILIES:
            candidates[f'{label}__{family}']=f[f'{family}_signed_bps']
    b.check(np.isfinite(candidates.to_numpy()).all() and np.isfinite(actual).all(),'Nonfinite forecasts/actuals')
    outputs=candidates.copy();outputs['side_means']=original.side_means_signed_bps;outputs['zero']=0.
    names={};ledger=[];selection_scores=[]
    for days in WINDOWS:
        pred,chosen,records,scores=monthly_select(candidates,actual,days,args.min_rows)
        outputs[f'monthly_{days}d']=pred;names[f'monthly_{days}d']=chosen
        ledger.extend(records);selection_scores.extend(scores)
    # One window is selected before the later section. Do not backfill its choice
    # into earlier rows, where that choice would depend on future outcomes.
    prior=(index+pd.Timedelta(hours=48)<freeze)
    b.check(prior.sum()>=args.min_rows,'Too few matured rows to select window')
    window_scores={days:float(((outputs.loc[prior,f'monthly_{days}d']-actual.loc[prior])**2).mean()) for days in WINDOWS}
    chosen_window=min(WINDOWS,key=lambda d:(window_scores[d],d))
    later=index>=freeze
    outputs['frozen_window_policy']=outputs[f'monthly_{chosen_window}d'].where(later)
    names['frozen_window_policy']=names[f'monthly_{chosen_window}d'].where(later,'unavailable_before_freeze')
    # Optimistic hindsight benchmark: actual section outcomes select its winner.
    oracle=pd.Series(np.nan,index=index);oracle_names=pd.Series('',index=index,dtype=object)
    oracle_ledger=[];section_scores=[]
    for issue,g in original.groupby('issue'):
        scores=((candidates.loc[g.index].sub(actual.loc[g.index],axis=0))**2).mean()
        winner=scores.idxmin()
        oracle.loc[g.index]=candidates.loc[g.index,winner];oracle_names.loc[g.index]=winner
        oracle_ledger.append(dict(issue=issue,model=winner,rows=len(g),rmse_bps=float(np.sqrt(scores[winner])),hindsight_only=True))
        for model,value in scores.items():
            section_scores.append(dict(issue=issue,model=model,rows=len(g),mse_bps2=float(value),rmse_bps=float(np.sqrt(value))))
    outputs['HINDSIGHT_quarter_winner']=oracle;names['HINDSIGHT_quarter_winner']=oracle_names
    out=b.EXP/'walkforward_model_selector_48h_v1'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=False)
    metrics=[];switches=[];long=[]
    for scope,mask in [('ALL',np.ones(len(index),dtype=bool)),('BEFORE_FREEZE',index<freeze),('AFTER_FREEZE',later)]:
        for model in outputs:
            # Never compare a partial-coverage policy to full-coverage competitors.
            if outputs.loc[mask,model].isna().any():continue
            truth=actual.loc[mask];pred=outputs.loc[mask,model]
            metrics.append(dict(scope=scope,model=model,**b.metric(truth,pred),
                direction_accuracy_pct=100*float((np.sign(pred)==np.sign(truth)).mean()),predicted_up_pct=100*float((pred>0).mean()),
                hindsight_only=model.startswith('HINDSIGHT')))
    for days in WINDOWS:
        rows=[r for r in ledger if r['window_days']==days]
        selected=[r['selected_model'] for r in rows]
        switches.append(dict(policy=f'monthly_{days}d',months=len(rows),
            switches=sum(a!=z for a,z in zip(selected,selected[1:])),
            fallback_months=sum(r['reason']=='insufficient_history_ridge_fallback' for r in rows)))
    frozen_rows=[r for r in ledger if r['window_days']==chosen_window and r['selection_time']>=freeze]
    frozen_names=[r['selected_model'] for r in frozen_rows]
    switches.append(dict(policy='frozen_window_policy',months=len(frozen_rows),
        switches=sum(a!=z for a,z in zip(frozen_names,frozen_names[1:])),
        fallback_months=sum(r['reason']=='insufficient_history_ridge_fallback' for r in frozen_rows)))
    for model in outputs:
        valid=outputs[model].notna();g=pd.DataFrame(index=index[valid]);g['model']=model
        g['chosen_model']=names[model].loc[g.index] if model in names else model
        g['predicted_return_pct']=outputs.loc[g.index,model]/100;g['actual_return_pct']=actual.loc[g.index]/100
        g['predicted_direction']=np.where(outputs.loc[g.index,model]>0,'UP',np.where(outputs.loc[g.index,model]<0,'DOWN','FLAT'))
        g['actual_direction']=np.where(actual.loc[g.index]>0,'UP',np.where(actual.loc[g.index]<0,'DOWN','FLAT'))
        g['direction_correct']=g.predicted_direction==g.actual_direction
        g['error_bps']=outputs.loc[g.index,model]-actual.loc[g.index];g['hindsight_only']=model.startswith('HINDSIGHT');long.append(g)
    table=pd.DataFrame(metrics);table.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(switches).to_csv(out/'switch_frequency.csv',index=False)
    pd.DataFrame(ledger).to_csv(out/'monthly_selection_ledger.csv',index=False)
    pd.DataFrame(selection_scores).to_csv(out/'monthly_candidate_scores.csv',index=False)
    pd.DataFrame(oracle_ledger).to_csv(out/'HINDSIGHT_quarter_winners.csv',index=False)
    pd.DataFrame(section_scores).to_csv(out/'quarter_model_scores.csv',index=False)
    outputs.insert(0,'actual_signed_return_bps',actual);outputs.to_csv(out/'all_predictions_bps.csv',index_label='decision_time')
    pd.concat(long).to_csv(out/'prediction_vs_actual_with_direction.csv',index_label='decision_time')
    b.dump(out/'FROZEN_WINDOW.json',dict(freeze_date=freeze,selected_window_days=chosen_window,
        selection_mse=window_scores,selection_rows=int(prior.sum()),max_selection_label_end=index[prior].max()+pd.Timedelta(hours=48)))
    b.dump(out/'CONTRACT.json',dict(candidates=list(candidates.columns),windows_days=WINDOWS,freeze_date=freeze,
        label_maturity='forecast_time+48h strictly before each monthly selection date',
        fallback='Ridge when the complete calendar lookback is not covered or fewer than min_rows eligible outcomes',
        min_rows=args.min_rows,selection='Lowest row-weighted matured signed-return MSE; ties resolved by fixed column order, Ridge first',
        unit='ordinary bps; percentages=bps/100',training=False,
        candidate_identity='Saved family/configuration streams with original quarterly refits; choosing a family does not freeze its underlying quarterly artifact',
        hindsight='Quarter winner uses actual section outcomes; optimistic benchmark over this finite catalog, not a deployable policy',
        limitation='Retrospectively constructed, historically exposed model catalog. Chronological selector simulation is not an untouched or historically available live strategy.',
        window_selection='Freeze one lookback using only matured pre-freeze policy outcomes; no pre-freeze backfill. Report all windows as diagnostics, not test-based winners.',
        input_hashes={str(q/'signed_return_predictions.csv'):b.sha(q/'signed_return_predictions.csv') for q in [args.tree_run,args.interaction_run]},
        source_sha256=b.sha(Path(__file__))))
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    show=['ridge','tree__extra_trees','interaction__extra_trees','side_means','monthly_30d','monthly_60d','monthly_90d','frozen_window_policy','HINDSIGHT_quarter_winner']
    summary=('WALK-FORWARD SAVED MODEL SELECTION - no training\n'
        f'Window frozen at {freeze}: {chosen_window} days, using earlier matured outcomes only.\n'
        'AFTER_FREEZE comparison uses identical rows. ALL is an additional historical diagnostic.\n'
        'HINDSIGHT uses future outcomes to choose each quarter winner; not an achievable selection rule.\n\n'
        +table[table.model.isin(show)&table.scope.isin(['ALL','AFTER_FREEZE'])].to_string(index=False)
        +'\n\n'+pd.DataFrame(switches).to_string(index=False))
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    b.dump(out/'COMPLETE.json',dict(status='COMPLETE',rows=len(index),training_performed=False,
        predictions_sha256=b.sha(out/'all_predictions_bps.csv'),contract_sha256=b.sha(out/'CONTRACT.json')))
    print(summary);print('Outputs saved:',out)


if __name__=='__main__':main()
