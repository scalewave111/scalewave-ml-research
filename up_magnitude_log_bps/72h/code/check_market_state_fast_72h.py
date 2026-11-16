"""Synthetic-only fast-mode size/configuration/replay benchmark."""
import time
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import market_state_trees_72h as m


def main():
    rng=np.random.default_rng(48)
    x=pd.DataFrame({f'STATE__synthetic{i}':rng.normal(size=8000) for i in range(241)})
    y=100+30*np.maximum(x.iloc[:,0].to_numpy(),0)
    m.configure(True)
    assert list(m.CONFIGS)==['hgb_fast'] and m.PACKS==['price_control','market_price']
    start=time.perf_counter()
    with threadpool_limits(limits=2):
        model=m.fit(x,y,dict(model='hgb_fast',pack='market_price'))
        assert model['estimator_training_rows']==6000 and model['available_training_rows']==8000
        assert model['model'].max_iter==60 and model['model'].max_bins==63
        assert model['median']==np.median(y)
        sample=x.iloc[-100:].copy();sample.iloc[0]=np.nan
        before=m.predict(model,sample,.5)
        assert before[0]==model['median']
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.joblib';joblib.dump(model,path)
            np.testing.assert_allclose(before,m.predict(joblib.load(path),sample,.5),rtol=0,atol=1e-10)
    print(f'PASS: fast training cap/config, full-training median, missing fallback/replay; one synthetic241feature fit+replay took {time.perf_counter()-start:.2f}s (not a full-run benchmark).')
    m.configure(False)
    assert m.TRAIN_ROW_CAP is None and 'extra_trees' in m.CONFIGS


if __name__=='__main__':main()
