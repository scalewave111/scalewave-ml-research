"""Diagnostics only: saved models, no new model fitting or changed selections."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import torch
import lstm_feature_architecture as m

OUT=m.OUT/'diagnostics_v1'
def save(name,rows):
    pd.DataFrame(rows).to_csv(OUT/name,index=False)
def load(s):
    return torch.load(m.path_for(s),weights_only=False,map_location='cpu')
def network(a):
    s=a['spec'];n=m.Network(a['inputs'],s['layers'],s['hidden'],s['dense'],a['median'])
    n.load_state_dict(a['state']);return n
def bootstrap(delta,keys):
    blocks=((keys-keys.min()).total_seconds()//(7*86400)).astype(int)
    d=pd.DataFrame(dict(delta=np.asarray(delta),block=blocks)).groupby('block').delta.agg(['sum','count'])
    ix=np.random.default_rng(48).integers(0,len(d),size=(2000,len(d)))
    samples=d['sum'].to_numpy()[ix].sum(1)/d['count'].to_numpy()[ix].sum(1)
    low,high=np.quantile(samples,[.025,.975])
    return dict(delta_mae=float(np.mean(delta)),ci_low=float(low),ci_high=float(high),
                includes_zero=bool(low<=0<=high),blocks=len(d))

def main():
    start=time.monotonic();m.initialize()
    OUT.mkdir(exist_ok=True)
    x,y,dev,ref,packs,checks=m.DATA
    complete=json.loads((m.OUT/'COMPLETE.json').read_text());assert complete['audit']['status']=='PASS'
    p=pd.read_csv(m.OUT/'predictions.csv',index_col=0,parse_dates=True)
    assert p.index.equals(dev)
    np.testing.assert_allclose(p.truth,y.loc[dev,'magnitude_log_bps'],atol=1e-8,rtol=0)
    np.testing.assert_allclose(p.baseline,ref.report_hybrid,atol=1e-8,rtol=0)
    decisions=json.loads((m.OUT/'ARCHITECTURE_SELECTIONS.json').read_text())
    features=json.loads((m.OUT/'FEATURE_SELECTIONS.json').read_text())
    windows=[];periods=[];uncertainty=[];frozen=[];refits=[];sequence=[]
    health=[];transfers=[];importance_rows=[];magnitudes=[];curves=[]
    diagnostic_ranks=[]
    for s,fs in zip(decisions,features):
        issue=pd.Timestamp(s['issue']);f=p[p.issue==s['issue']];te=f.index
        feature_start=issue-pd.DateOffset(months=3);inner_origin=issue-pd.DateOffset(months=1)
        inner=load(s['selected']['spec']);outer=load(s['outer_spec'])
        keys=inner['valid_keys'];truth_val=y.loc[keys,'magnitude_log_bps'].to_numpy()
        truth=f.truth.to_numpy()
        for name in ('baseline','previous_lstm','selected_lstm'):
            err=f[name]-f.truth
            windows.append(dict(issue=s['issue'],model=name,rows=len(f),
                mae=float(abs(err).mean()),mean_error=float(err.mean()),median_error=float(err.median()),
                forecast_median=float(f[name].median()),forecast_std=float(f[name].std()),
                truth_median=float(f.truth.median()),truth_std=float(f.truth.std()),
                contribution_to_total_gap=float((abs(err)-abs(f.baseline-f.truth)).sum()/len(p))))
        populations={
            'feature_selection':y.index[(y.index>=feature_start)&(y.index<inner_origin)&(y.label_end<inner_origin)],
            'architecture_training':inner['train_keys'],
            'architecture_validation':keys,'outer_training':outer['train_keys'],'outer_test':te}
        for phase,k in populations.items():
            v=y.loc[k,'magnitude_log_bps']
            periods.append(dict(issue=s['issue'],phase=phase,rows=len(k),median=float(v.median()),
                mean=float(v.mean()),p10=float(v.quantile(.1)),p90=float(v.quantile(.9))))
        for stage,records in [('feature_count',fs['count_trials']),('architecture',s['trials'])]:
            ordered=sorted(records,key=m.rank);best,runner=ordered[:2]
            aa,bb=load(best['spec']),load(runner['spec'])
            assert aa['valid_keys'].equals(bb['valid_keys'])
            target=y.loc[aa['valid_keys'],'magnitude_log_bps'].to_numpy()
            delta=abs(aa['validation_prediction']-target)-abs(bb['validation_prediction']-target)
            uncertainty.append(dict(issue=s['issue'],stage=stage,winner=best['tag'],
                runner_up=runner['tag'],**bootstrap(delta,aa['valid_keys'])))
        # Exposed-test diagnostic, not a replacement model-selection procedure.
        candidates={r['tag']:r for r in fs['count_trials']+s['trials']}
        later_predictions={}
        arch_tags={r['tag'] for r in s['trials']}
        for tag,r in candidates.items():
            aa=load(r['spec']);pred=m.prediction(aa,te);later_predictions[tag]=pred
            frozen.append(dict(issue=s['issue'],tag=tag,architecture_candidate=tag in arch_tags,
                selected=tag==s['selected']['tag'],train_origin=str(aa['origin']),
                validation_mae=r['validation_mae'],diagnostic_later_mae=float(abs(pred-truth).mean())))
        arch_order=sorted(arch_tags,key=lambda t:float(abs(later_predictions[t]-truth).mean()))
        diagnostic_ranks.append(dict(issue=s['issue'],validation_winner=s['selected']['tag'],
            later_rank_among_six=arch_order.index(s['selected']['tag'])+1,
            hindsight_best_frozen_inner=arch_order[0],
            warning='All models here were trained one month before the deployed outer refit; hindsight diagnostics only'))
        pred_inner=later_predictions[s['selected']['tag']]
        pred_outer=m.prediction(outer,te)
        np.testing.assert_allclose(pred_outer,f.selected_lstm,atol=1e-8,rtol=0)
        refits.append(dict(issue=s['issue'],inner_frozen_mae=float(abs(pred_inner-truth).mean()),
            outer_refit_mae=float(abs(pred_outer-truth).mean()),
            forecast_shift=float(np.mean(pred_outer-pred_inner)),
            **bootstrap(abs(pred_outer-truth)-abs(pred_inner-truth),te)))
        for phase,a,k,target in [('validation',inner,keys,truth_val),('test',outer,te,truth)]:
            net=network(a);tx=m.q.seq(x,k,a['prep'],a['sequence']);original=m.infer(net,tx)
            for label,z in [('reverse_past_keep_current',tx[:,list(range(tx.shape[1]-2,-1,-1))+[tx.shape[1]-1],:]),
                            ('repeat_current',tx[:,-1:,:].expand(-1,tx.shape[1],-1).contiguous())]:
                altered=m.infer(net,z)
                sequence.append(dict(issue=s['issue'],phase=phase,intervention=label,
                    mean_prediction_change=float(abs(altered-original).mean()),
                    original_mae=float(abs(original-target).mean()),altered_mae=float(abs(altered-target).mean()),
                    delta_mae=float(abs(altered-target).mean()-abs(original-target).mean())))
        # Compare the old reference-model ranking with the newer full-input control.
        fullspec=m.spec(issue,fs['ranked_columns'],(1,8,0));full=load(fullspec)
        cols=full['prep']['columns'];tx=m.q.seq(x,full['valid_keys'],full['prep'],full['sequence'])
        net=network(full);target=y.loc[full['valid_keys'],'magnitude_log_bps'].to_numpy()
        base=abs(m.infer(net,tx)-target)
        chunks=[np.arange(i,min(i+336,len(tx))) for i in range(0,len(tx),336)]
        permutations=[]
        for seed in (48,149,250):
            order=np.random.default_rng(seed).permutation(len(chunks))
            if np.array_equal(order,np.arange(len(chunks))):order=np.roll(order,1)
            permutations.append(np.concatenate([chunks[j] for j in order]))
        new_importance={}
        for j,col in enumerate(cols):
            changes=[]
            for perm in permutations:
                altered=tx.clone();altered[:,:,j]=tx[perm,:,j]
                altered[:,:,j+len(cols)]=tx[perm,:,j+len(cols)]
                changes.append(float((abs(m.infer(net,altered)-target)-base).mean()))
            new_importance[col]=float(np.mean(changes))
            train=x.loc[full['train_keys'],col];valid=x.loc[full['valid_keys'],col];test=x.loc[te,col]
            scale=full['prep']['scale'][col];med=full['prep']['median'][col]
            health.append(dict(issue=s['issue'],feature=col,
                train_unique=int(train.nunique()),train_missing=float(train.isna().mean()),
                validation_missing=float(valid.isna().mean()),test_missing=float(test.isna().mean()),
                test_clipped_fraction=float((((test.fillna(med)-med)/scale).abs()>8).mean())))
        old=pd.read_csv(m.OUT/'features'/issue.strftime('%Y%m%d')/'importance.csv').set_index('feature').mae_increase_bps
        new=pd.Series(new_importance).reindex(old.index)
        old_top=set(old.nlargest(4).index);new_top=set(new.nlargest(4).index)
        transfers.append(dict(issue=s['issue'],rank_correlation=float(old.corr(new,method='spearman')),
            top4_overlap=len(old_top&new_top),old_top4=list(old.nlargest(4).index),later_top4=list(new.nlargest(4).index)))
        for col in old.index:
            importance_rows.append(dict(issue=s['issue'],feature=col,earlier_reference_importance=float(old[col]),
                later_control_importance=float(new[col]),selected=col in s['features']))
        training_y=y.loc[outer['train_keys'],'magnitude_log_bps']
        median,p90=training_y.quantile([.5,.9])
        labels=np.where(truth<=median,'low: at/below training median',np.where(truth<=p90,'middle: training median to p90','high: above training p90'))
        for label in sorted(set(labels)):
            keep=labels==label
            magnitudes.append(dict(issue=s['issue'],group=label,rows=int(keep.sum()),
                lstm_mae=float(abs(pred_outer[keep]-truth[keep]).mean()),
                baseline_mae=float(abs(f.baseline.to_numpy()[keep]-truth[keep]).mean()),
                lstm_median_error=float(np.median(pred_outer[keep]-truth[keep])),
                contribution_to_total_gap=float((abs(pred_outer[keep]-truth[keep])-abs(f.baseline.to_numpy()[keep]-truth[keep])).sum()/len(p))))
        for path in (m.OUT/'models'/issue.strftime('%Y%m%d')).glob('inner_*.pt'):
            aa=torch.load(path,weights_only=False,map_location='cpu')
            h=aa['history'];best=h[aa['epoch']-1]
            curves.append(dict(issue=s['issue'],tag=aa['spec']['tag'],
                selected=aa['spec']['tag']==s['selected']['tag'],best_epoch=aa['epoch'],epochs_run=len(h),
                train_first=h[0]['train_mae'],train_last=h[-1]['train_mae'],
                validation_first=h[0]['validation_mae'],validation_best=best['validation_mae'],
                validation_last=h[-1]['validation_mae']))
        print('DIAGNOSTICS WINDOW COMPLETE',issue.date(),flush=True)
    for name,rows in [('window_diagnostics.csv',windows),('period_shift.csv',periods),
        ('selection_uncertainty.csv',uncertainty),('frozen_inner_models_hindsight.csv',frozen),
        ('refit_diagnostics.csv',refits),('sequence_diagnostics.csv',sequence),
        ('feature_health.csv',health),('importance_transfer.csv',importance_rows),
        ('feature_rank_transfer.csv',transfers),('magnitude_errors.csv',magnitudes),
        ('learning_curves.csv',curves),('validation_choice_later_rank.csv',diagnostic_ranks)]:
        save(name,rows)
    selected_windows=[r for r in windows if r['model']=='selected_lstm']
    summary=dict(no_new_training=True,elapsed_seconds=time.monotonic()-start,
        window_errors=selected_windows,selection_uncertainty=uncertainty,
        refit_diagnostics=refits,rank_transfer=transfers,validation_choice_later_rank=diagnostic_ranks,
        inactive_inputs=[r for r in health if r['train_unique']<=1],
        sequence_diagnostics=sequence,
        limitations='Hindsight diagnostics on exposed test data; no replacement selections made. Bootstrap intervals use few calendar blocks and do not correct for model selection. Sequence interventions alter the input distribution.')
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2,default=str,allow_nan=False),encoding='utf-8')
    report='LSTM DIAGNOSTICS — NO NEW TRAINING\\n\\n'
    for title,rows in [('Window errors',selected_windows),('Selection uncertainty',uncertainty),
                       ('Target levels by period',periods),('Refit changes',refits),
                       ('Sequence interventions',sequence),('Feature-rank transfer',transfers)]:
        report+=title+'\\n'+pd.DataFrame(rows).to_string(index=False)+'\\n\\n'
    report+=summary['limitations']+'\\n'
    (OUT/'DIAGNOSTICS.txt').write_text(report,encoding='utf-8')
    print(json.dumps(summary,indent=2,default=str),flush=True)

if __name__=='__main__':main()

