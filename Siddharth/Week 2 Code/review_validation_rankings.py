"""Read-only validation-to-test ranking review across completed experiments.
Save in Week 2/Code. Reads CSV/JSON only; no fitting, tuning or model replay.
Dependencies: numpy pandas scipy. No torch or prepared dataset required.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

EXPERIMENTS={
 'tabular_mae_candidates_v1':('tabular',None),
 'mlp_variants_mae_v1':('standard','architecture'),
 'training_windows_mae_v1':('standard','window'),
 'mlp_regularization_mae_v1':('standard','setting'),
 'mlp_linear_alpha_mae_v1':('standard','setting'),
 'spline_median_mae_v1':('standard','model')}


def check(ok,msg):
    if not ok: raise ValueError(msg)


def normalize(frame):
    f=frame.copy();f['issue']=pd.to_datetime(f.issue,utc=True).astype(str)
    check(not f.duplicated(['issue','side','model']).any(),'Duplicate candidate rows')
    return f


def assess(v,t,selected):
    check(set(v.model)==set(t.model),'Validation/test candidate sets differ')
    g=v.merge(t,on='model',suffixes=('_validation','_test'),validate='one_to_one')
    check(len(g)>=2 and g.mae_bps_validation.notna().all() and g.mae_bps_test.notna().all(),'Invalid candidate scores')
    check(np.isfinite(g[['mae_bps_validation','mae_bps_test']].to_numpy(float)).all(),'Nonfinite MAE')
    check(g.rows_validation.nunique()==1 and g.rows_test.nunique()==1,'Candidate row counts differ within stage')
    check(selected in set(g.model),'Selected candidate missing')
    chosen=g[g.model.eq(selected)].iloc[0]
    check(chosen.mae_bps_validation<=g.mae_bps_validation.min()+1e-8,'Saved choice is not a validation-MAE minimum')
    g['validation_mae_rank']=g.mae_bps_validation.rank(method='average')
    g['test_mae_rank']=g.mae_bps_test.rank(method='average')
    g['selected_on_validation']=g.model.eq(selected)
    rho=g.mae_bps_validation.corr(g.mae_bps_test,method='spearman') if g.mae_bps_validation.nunique()>1 and g.mae_bps_test.nunique()>1 else np.nan
    gap=float(chosen.mae_bps_test-g.mae_bps_test.min())
    others=g.loc[~g.model.eq(selected),'mae_bps_validation']
    note=dict(candidates=len(g),selected_model=selected,validation_rows=int(chosen.rows_validation),test_rows=int(chosen.rows_test),
        validation_winner_margin_bps=float(others.min()-chosen.mae_bps_validation),
        selected_test_mae_bps=float(chosen.mae_bps_test),
        hindsight_best_test_mae_bps=float(g.mae_bps_test.min()),
        hindsight_best_test_models=' | '.join(g.loc[np.isclose(g.mae_bps_test,g.mae_bps_test.min(),atol=1e-8,rtol=0),'model']),
        selected_test_rank=float(g.loc[g.model.eq(selected),'test_mae_rank'].iloc[0]),
        selected_is_test_best=gap<=1e-8,test_gap_to_hindsight_best_bps=max(gap,0.),
        rank_agreement_spearman=float(rho) if pd.notna(rho) else np.nan,
        test_candidate_mae_spread_bps=float(g.mae_bps_test.max()-g.mae_bps_test.min()))
    return g,note


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--out',type=Path)
    args=ap.parse_args();ws=args.workspace.resolve();root=ws/'Code/experiments'
    out=args.out or root/'validation_ranking_review_v1'
    check(not out.exists(),f'Preserving {out}; use --out with a new folder')
    details=[];folds=[];inventory=[]
    for name,(kind,key) in EXPERIMENTS.items():
        folder=root/name
        if not folder.exists():
            inventory.append(dict(experiment=name,status='SKIPPED',reason='Folder absent'));continue
        try:
            decisions=json.loads((folder/'SELECTIONS.json').read_text(encoding='utf-8-sig'))
            scores=pd.read_csv(folder/'scores.csv')
            tests=normalize(scores[scores.issue.ne('POOLED')])
            vals=[];choices=[]
            if kind=='tabular':
                # Family scores must match the selected configurations actually refit.
                # Raw component trials have no matching test fits for every setting.
                for d in decisions:
                    for family,component in [('median_linear','linear'),('boosting','boost'),('median_boost_blend','blend')]:
                        r=d[component]
                        vals.append(dict(issue=d['issue'],side=d['side'],model=family,
                                         **{k:r[k] for k in ['rows','mae_bps','rank_ic']}))
                    choices.append(dict(issue=d['issue'],side=d['side'],selected=d['selected_family']))
                val=normalize(pd.DataFrame(vals))
            else:
                val=pd.read_csv(folder/'validation_scores.csv')
                if key!='model': val=val.rename(columns={key:'model'})
                val=normalize(val)
                choices=[dict(issue=d['issue'],side=d['side'],selected=d[key]) for d in decisions]
            choice=pd.DataFrame(choices);choice['issue']=pd.to_datetime(choice.issue,utc=True).astype(str)
            check(not choice.duplicated(['issue','side']).any(),'Duplicate selections')
            check(set(zip(choice.issue,choice.side))==set(zip(val.issue,val.side)),'Missing fold selections')
            local_details=[];local_folds=[]
            for c in choice.to_dict('records'):
                v=val[(val.issue==c['issue'])&(val.side==c['side'])]
                t=tests[(tests.issue==c['issue'])&(tests.side==c['side'])&tests.model.isin(v.model)]
                cols=['model','rows','mae_bps','rank_ic']
                g,note=assess(v[cols],t[cols],c['selected'])
                selected_score=tests[(tests.issue==c['issue'])&(tests.side==c['side'])&tests.model.eq('validation_selected')]
                check(len(selected_score)==1,'Missing validation-selected test score')
                check(abs(float(selected_score.iloc[0].mae_bps)-note['selected_test_mae_bps'])<=1e-8,'Selected test MAE mismatch')
                refs=tests[(tests.issue==c['issue'])&(tests.side==c['side'])]
                refname='existing_reference' if name=='training_windows_mae_v1' else ('existing_mlp' if c['side']=='UP' else 'existing_hybrid')
                r=refs[refs.model.eq(refname)];check(len(r)==1,'Missing reference model')
                note['selected_gain_vs_reference_bps']=float(r.iloc[0].mae_bps)-note['selected_test_mae_bps']
                for k,value in [('experiment',name),('issue',c['issue']),('side',c['side'])]:
                    g[k]=value;note[k]=value
                local_details.append(g);local_folds.append(note)
            details.extend(local_details);folds.extend(local_folds)
            inventory.append(dict(experiment=name,status='REVIEWED',reason='',fold_side_comparisons=len(local_folds)))
            print(f'Reviewed {name}',flush=True)
        except Exception as e:
            # Never silently compare partial or incompatible experiment files.
            inventory.append(dict(experiment=name,status='FAILED',reason=f'{type(e).__name__}: {e}'))
            print(f'Could not review {name}: {e}',flush=True)
    check(len(folds)>0,'No valid comparisons. Check existing experiment CSV/JSON files.')
    fold=pd.DataFrame(folds);aggregate=[]
    for (experiment,side),g in fold.groupby(['experiment','side'],sort=False):
        aggregate.append(dict(experiment=experiment,side=side,folds=len(g),
            validation_winner_also_test_best=int(g.selected_is_test_best.sum()),
            defined_rank_agreement_folds=int(g.rank_agreement_spearman.notna().sum()),
            median_rank_agreement=g.rank_agreement_spearman.median(),
            weighted_gap_to_hindsight_best_bps=float(np.average(g.test_gap_to_hindsight_best_bps,weights=g.test_rows)),
            weighted_selected_gain_vs_reference_bps=float(np.average(g.selected_gain_vs_reference_bps,weights=g.test_rows))))
    agg=pd.DataFrame(aggregate)
    summary=('READ-ONLY VALIDATION RANKING REVIEW\n'
        'Rank agreement compares candidate MAE rankings, NOT forecast Rank IC. +1=same order; -1=reversed.\n'
        'Positive selected gain versus reference means selection improved test MAE.\n'
        'Gap to hindsight best is descriptive and optimistic: that test winner was unknowable at selection time.\n'
        'Near ties can reorder ranks with negligible MAE changes; inspect spreads and winner margins.\n'
        'Validation uses inner fits; tests use outer refits and different dates/populations. Rank changes do not establish a cause.\n'
        'Tabular review compares selected families only; component settings without saved test results are not compared.\n'
        'These experiments reuse the same four exposed periods; counts across experiments are not independent evidence.\n'
        'No significance tests, new selection, fitting, or proposed changes to the validation design.\n\n'
        +agg.to_string(index=False)+'\n\nPER-FOLD COMPARISONS\n'+fold.to_string(index=False)
        +'\n\nINPUT STATUS\n'+pd.DataFrame(inventory).to_string(index=False))
    out.mkdir(parents=True)
    agg.to_csv(out/'experiment_summary.csv',index=False);fold.to_csv(out/'fold_comparisons.csv',index=False)
    pd.concat(details,ignore_index=True).to_csv(out/'candidate_rankings.csv',index=False)
    pd.DataFrame(inventory).to_csv(out/'input_status.csv',index=False)
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (out/'COMPLETE.json').write_text(json.dumps(dict(status='PARTIAL' if any(r['status']=='FAILED' for r in inventory) else 'COMPLETED',fold_side_comparisons=len(fold))),encoding='utf-8')
    print(summary);print(f'Outputs: {out}')


if __name__=='__main__': main()
