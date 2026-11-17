"""Read-only model replay and independent selection checks; writes audit/results only."""
import json,itertools
import numpy as np
import pandas as pd
import joblib
import eight_feature_search as e
s=e.s
def run():
    x,y,dev,ref=joblib.load(e.OUT/'prepared.joblib');rows=[];selections=[]
    expected={s.key(c) for n in range(1,9) for c in itertools.combinations(e.COLS,n)}
    for method in ['rf','mlp']:
        for issue in s.q.STARTS:
            f=e.OUT/method/f'{issue:%Y%m%d}';trials=json.loads((f/'screen_trials.json').read_text())
            candidates=json.loads((f/'FOCUSED_CANDIDATES.json').read_text())
            assert {r['id'] for r in trials}=={s.key(c) for c in candidates}
            assert {r['id'] for r in trials}.issubset(expected)
            full=json.loads((f/'FULL_VALIDATION.json').read_text());best=json.loads((f/'SELECTION.json').read_text())
            assert best==min(full,key=lambda r:(r['mae'],len(r['cols']),r['id']))
            inner,ik,vk,ot,te=s.q.windows(y,dev,issue);e.configure(method,best['config'])
            ai=joblib.load(f/'full'/best['id']/'inner/bundle.joblib');ao=joblib.load(f/'full'/best['id']/'outer/bundle.joblib')
            assert ai['keys'].equals(ik) and ao['keys'].equals(ot)
            assert ai['cutoff']==inner and ao['cutoff']==issue
            assert y.loc[ik,'label_end'].max()<inner and y.loc[vk,'label_end'].max()<issue and y.loc[ot,'label_end'].max()<issue
            assert set(best['cols']).issubset(e.COLS)
            if method=='rf':assert ai['weight']==ao['weight']
            else:
                confirmed=joblib.load(f/'confirmation'/best['id']/'inner/bundle.joblib')
                assert confirmed['epochs']==ao['epochs'] and len(ao['epochs'])==3
            assert abs(np.mean(abs(s.predict(ai,x,vk)-y.loc[vk,'magnitude_log_bps']))-best['mae'])<1e-8
            saved=pd.read_csv(f/'predictions.csv',index_col=0,parse_dates=True);assert saved.index.equals(te)
            np.testing.assert_allclose(saved.selected,s.predict(ao,x,te),atol=1e-8,rtol=0)
            for c in best['cols']:selections.append(dict(method=method,issue=str(issue),feature=c,config=json.dumps(best['config'])))
    manifest=json.loads((e.OUT/'MANIFEST.json').read_text())
    assert all(s.q.sha(path)==digest for path,digest in manifest['inputs'].items())
    p=pd.read_csv(e.OUT/'predictions.csv',index_col=0,parse_dates=True);assert p.index.equals(dev)
    np.testing.assert_allclose(p.truth,y.loc[dev,'magnitude_log_bps'],atol=1e-8,rtol=0)
    np.testing.assert_allclose(p.baseline,ref.report_hybrid,atol=1e-8,rtol=0)
    for method in ['rf','mlp']:
        for reference in ['baseline','yesterday_best']:
            d=abs(p[method]-p.truth)-abs(p[reference]-p.truth)
            blocks=(p.index.asi8-p.index.asi8.min())//pd.Timedelta(days=7).value
            b=pd.DataFrame(dict(delta=d.to_numpy(),block=blocks)).groupby('block').delta.agg(['sum','count'])
            ids=np.random.default_rng(20261105).integers(0,len(b),(5000,len(b)))
            boot=b['sum'].to_numpy()[ids].sum(1)/b['count'].to_numpy()[ids].sum(1)
            lo,hi=np.quantile(boot,[.025,.975]);rows.append(dict(model=method,reference=reference,mae_difference=d.mean(),lower=lo,upper=hi))
    pd.DataFrame(rows).to_csv(e.OUT/'uncertainty.csv',index=False)
    pd.DataFrame(selections).to_csv(e.OUT/'selected_features.csv',index=False)
    total=sum(len(json.loads(p.read_text())) for p in e.OUT.glob('*/*/screen_trials.json'))
    s.dump(e.OUT/'INDEPENDENT_AUDIT.json',dict(status='PASS',subsets=total,rows=len(dev),all_registered_focused_candidates_present=True,exhaustive=False,selection_argmin=True,purges=True,checkpoint_replay=True,inputs_unchanged=True))
    print('AUDIT PASS',flush=True)
if __name__=='__main__':run()
