"""Synthetic-only formula, selection, fit/replay, pipeline and resume checks."""
from pathlib import Path
import json
import sys
import tempfile
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import tune_excursions_15m as s


def main():
    rng = np.random.default_rng(48)
    ix = pd.date_range('2023-01-01', periods=200, freq='5min', tz='UTC')
    r = pd.Series(rng.normal(size=len(ix)), index=ix)
    origin = ix[100]
    f = s.short_features(r, pd.DatetimeIndex([origin]))
    assert len(f.columns) == 16
    for minutes in [15,30]:
        raw = r[(r.index >= origin-pd.Timedelta(minutes=minutes)) & (r.index < origin)]
        path = np.r_[0., raw.cumsum().to_numpy()]
        expected = dict(rv=np.sqrt((raw**2).sum()), up_rv=np.sqrt((raw.clip(lower=0)**2).sum()),
            down_rv=np.sqrt((raw.clip(upper=0)**2).sum()), net=raw.sum(), past_range=path.max()-path.min(),
            distance_from_high=path.max()-path[-1], distance_from_low=path[-1]-path.min(), max_abs_return=raw.abs().max())
        for name,value in expected.items():
            np.testing.assert_allclose(f.iloc[0][f'SHORTPRICE__{name}__{minutes}m'],value,atol=1e-10)
    s.check_causal(r,ix[30:150])
    r.loc[ix[99]] = np.nan
    assert s.short_features(r,pd.DatetimeIndex([origin])).isna().all().all()
    truth = [np.ones(50)*10 for _ in range(3)]
    old = [np.ones(50)*12 for _ in range(3)]
    winner, _ = s.choose(truth, {'signal':truth}, old, old, [50]*3,.001)
    assert winner['candidate'] == 'signal' and winner['weight'] == 1
    fallback, _ = s.choose(truth, {'bad':[np.zeros(50)]*3}, old, old, [50]*3,.001)
    assert fallback['candidate'] == 'current' and fallback['weight'] == 0
    zero, _ = s.choose([np.zeros(50)]*3, {'zero':[np.zeros(50)]*3}, [np.zeros(50)]*3,[np.zeros(50)]*3,[50]*3,.001)
    assert zero['weight'] == 0
    # Complete synthetic execution, including saved-model replay and resume hashes.
    issue = pd.Timestamp('2024-01-01',tz='UTC')
    index = pd.date_range(end=issue-pd.Timedelta(minutes=30), periods=3600,freq='30min',tz='UTC').append(pd.date_range(issue,periods=120,freq='30min'))
    x = pd.DataFrame({'signal':rng.uniform(size=len(index)), 'noise':rng.normal(size=len(index)),
        'short':rng.uniform(size=len(index))},index=index)
    y = pd.DataFrame({t:10+20*x.signal+5*x.short+rng.normal(size=len(index)) for t in s.base.TARGETS},index=index)
    folds = [(np.arange(n),np.arange(n,n+200)) for n in [1200,2000,2800]]
    cards, frames = {},[]
    with threadpool_limits(limits=2), tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for target in s.base.TARGETS:
            yy=y[target].to_numpy()
            current=s.fit(x[['signal']].to_numpy()[:3600],yy[:3600],'unchanged')
            original=s.base.fit(x[['signal']].to_numpy()[:3600],yy[:3600],'tree')
            np.testing.assert_array_equal(s.base.predict(current,x[['signal']].to_numpy()[3600:]),s.base.predict(original,x[['signal']].to_numpy()[3600:]))
            wide=s.fit(x[['signal','noise']].to_numpy()[:3600],yy[:3600],'unchanged')
            errors=[]
            for tr,va in folds:
                a=x[['signal']].to_numpy()
                errors.append(float(np.abs(yy[va]-s.base.predict(s.fit(a[tr],yy[tr],'unchanged'),a[va])).mean()))
            cards[(issue,target)]={'selected_variant':'original','variants':{
                'original':dict(features=['signal'],validation_window_mae=errors),
                'all_price_plus_original':dict(features=['signal','noise'])}}
            frames.append(pd.DataFrame(dict(decision_time=index[3600:],issue=issue,target=target,actual=yy[3600:],
                selected=s.base.predict(current,x[['signal']].to_numpy()[3600:]),
                all_price_plus_original=s.base.predict(wide,x[['signal','noise']].to_numpy()[3600:]))))
        old=pd.concat(frames,ignore_index=True)
        packs=s.feature_packs(x,folds,cards[(issue,s.base.TARGETS[0])],['short'])
        assert packs['current_features']==['signal'] and packs['all_price_short']==['signal','noise','short']
        changed=x.copy();changed.iloc[3600:]=99999
        assert s.feature_packs(changed,folds,cards[(issue,s.base.TARGETS[0])],['short']) == packs
        s.inputs=lambda args:(x,y,index[3600:],old,cards,['short'],[Path(s.__file__)])
        s.base.ISSUES=pd.DatetimeIndex([issue])
        s.base.windows=lambda index,issue:folds
        s.CONFIGS={k:v for k,v in s.CONFIGS.items() if k in ['unchanged','smaller_leaf']}
        sys.argv=['tune','--output-dir',str(root),'--threads','2']
        s.main()
        run=next(root.glob('*/COMPLETE.json')).parent
        receipt=json.loads((run/'COMPLETE.json').read_text())
        assert receipt['replay_passed'] and receipt['prediction_rows']==240
        before=pd.read_csv(run/'predictions.csv')
        s.fit=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Resume unexpectedly fitted a model'))
        sys.argv=['tune','--resume',str(run),'--threads','2']
        s.main()
        after=pd.read_csv(run/'predictions.csv')
        np.testing.assert_allclose(before.select_dtypes('number'),after.select_dtypes('number'),rtol=0,atol=1e-10)
        checkpoint=next(run.glob('*/predictions.csv'))
        with checkpoint.open('a') as stream:stream.write('\n')
        try:s.main()
        except ValueError as error:assert 'Checkpoint hash mismatch' in str(error)
        else:raise AssertionError('Corrupted checkpoint accepted')
    print('PASS: short-price formulas/causality/missingness; gates/fallback; old control reproduction; pack independence; full synthetic pipeline/replay; resume and corruption rejection.')


if __name__=='__main__':main()
