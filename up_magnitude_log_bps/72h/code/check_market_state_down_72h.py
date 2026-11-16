"""Synthetic-only formula, causality, joint model and output checks."""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import market_state_features_72h as f
import market_state_trees_72h as m
import fit_market_state_down_72h as runner


def main():
    rng=np.random.default_rng(48)
    dates=pd.date_range('2022-01-01',periods=2000,freq='30min',tz='UTC')
    market=pd.DataFrame({c:rng.uniform(0,1,len(dates)) for c in f.SOURCES.values()},index=dates)
    price=pd.DataFrame({f'LONG__{name}__{h}h':rng.normal(size=len(dates)) for name in ['normalized_momentum','vol_per_sqrt_hour'] for h in [24,72,168]},index=dates)
    state,dictionary=f.build(market,price,dates)
    assert len(state.columns)==205 and dictionary.feature.is_unique
    cut=dates[1500];source=f.SOURCES['funding_level'];past=market.loc[(dates>=cut-pd.Timedelta(hours=24))&(dates<cut),source]
    np.testing.assert_allclose(state.loc[cut,'STATE__funding_level_mean__24h'],past.mean())
    np.testing.assert_allclose(state.loc[cut,'STATE__funding_level_std__24h'],past.std(ddof=0))
    np.testing.assert_allclose(state.loc[cut,'STATE__funding_level_deviation__24h'],market.loc[cut-pd.Timedelta(minutes=30),source]-past.mean())
    np.testing.assert_allclose(state.loc[cut,'STATE__funding_level_high_share__24h'],past.ge(.8).mean())
    np.testing.assert_allclose(state.loc[cut,'INTERACT__funding_oi__24h'],state.loc[cut,'STATE__funding_level_mean__24h']*state.loc[cut,'STATE__oi_position_mean__24h'])
    changed=market.copy();changed.loc[dates>=cut]=1e6
    keys=dates[1490:1501]
    pd.testing.assert_frame_equal(state.loc[keys],f.build(changed,price.loc[keys],keys)[0])
    pd.testing.assert_frame_equal(state.loc[keys],f.build(market.loc[dates<=cut],price.loc[keys],keys)[0])
    missing=market.copy();missing.loc[dates[1496:1500],source]=np.nan
    assert np.isnan(f.build(missing,price.loc[[cut]],pd.DatetimeIndex([cut]))[0].loc[cut,'STATE__funding_level_mean__6h'])
    xx=pd.DataFrame({'LONG__a__24h':rng.uniform(size=3600),'STATE__b':rng.uniform(size=3600)})
    xx['INTERACT__ab']=xx.iloc[:,0]*xx.iloc[:,1]
    y=20+80*xx.INTERACT__ab.to_numpy()
    folds=[(np.arange(k),np.arange(k,k+200)) for k in [1200,1600,2000,2400]]
    with tempfile.TemporaryDirectory() as temp,threadpool_limits(limits=2):
        root=Path(temp)
        for kind in m.CONFIGS:
            model=m.fit(xx.iloc[:1600],y[:1600],dict(model=kind,pack='market_price'))
            assert len(model['features'])==3
            pred=m.predict(model,xx.iloc[-20:],.5)
            joblib.dump(model,root/'model.joblib')
            np.testing.assert_allclose(pred,m.predict(joblib.load(root/'model.joblib'),xx.iloc[-20:],.5),rtol=0,atol=1e-10)
            blank=xx.iloc[-20:].copy();blank.iloc[0]=np.nan
            assert m.predict(model,blank,.5)[0]==model['median']
            np.testing.assert_allclose(pred,.5*m.predict(model,xx.iloc[-20:])+.5*model['median'])
        for name in ['a','b','zero']:(root/name).mkdir()
        with patch.object(m,'CONFIGS',{'hgb_regularized':m.CONFIGS['hgb_regularized']}),patch.object(m,'PACKS',['market_price']):
            choice=m.search(xx,y,folds,root/'a');assert choice['model']!='median'
            changed=xx.copy();changed.iloc[2600:]=1e6;yy=y.copy();yy[2600:]=1e6
            assert choice==m.search(changed,yy,folds,root/'b')
            assert m.search(xx,np.zeros(len(y)),folds,root/'zero')['model']=='median'
    index=pd.date_range('2021-01-01','2025-01-22',freq='6h',tz='UTC')
    x=pd.DataFrame({'LONG__a__24h':rng.uniform(size=len(index)),'STATE__b':rng.uniform(size=len(index))},index=index)
    x['INTERACT__ab']=x.iloc[:,0]*x.iloc[:,1];y=20+80*x.INTERACT__ab.to_numpy()
    x.attrs['excluded_labels']=pd.DataFrame(columns=['decision_time','label_status'])
    dev=index[index>=m.b.ISSUES[0]]
    for issue in m.b.ISSUES:
        for tr,va in runner.windows(index,issue):
            assert (index[tr]+pd.Timedelta(hours=72)<index[va].min()).all()
            assert (index[va]+pd.Timedelta(hours=72)<issue).all()
    old=pd.DataFrame([dict(decision_time=t,training_median=np.median(y[index+pd.Timedelta(hours=72)<m.b.ISSUES[m.b.ISSUES<=t][-1]]),
        previous_neural=50.,previous_transformed=50.,previous_guarded=50.,previous_anchor=50.) for t in dev])
    def load(args):return x,y,dev,old,pd.DataFrame([dict(feature=c,formula='synthetic') for c in x]),[]
    def choose(*args):return dict(model='hgb_regularized',pack='market_price',weight=.5,validation_mae=1.)
    with tempfile.TemporaryDirectory() as temp,threadpool_limits(limits=2):
        with patch.object(runner,'load',load),patch.object(m,'search',choose),patch.object(sys,'argv',['runner','--output-dir',temp]):runner.main()
        out=list(Path(temp).iterdir())[0];done=json.loads((out/'COMPLETE.json').read_text())
        assert done['replay_passed'] and done['development_origins']==len(dev)
        assert done['predictions_sha256']==m.b.sha(out/'predictions.csv')
    print('PASS: snapshot formulas/interactions, strict causality/prefix/missingness, alltree models, joint signal, train-only selection, gates/fallback/blending/replay,72hpurges and full pipeline.')


if __name__=='__main__':main()
