"""Synthetic calibration causality, maturity, selection and fallback tests."""
import numpy as np
import pandas as pd
import improve_best_models_72h as r


def main():
    times=pd.date_range('2024-01-01','2025-01-20',freq='30min',tz='UTC')
    frame=pd.DataFrame(dict(decision_time=times,actual=120.,best_model=100.,training_median=100.))
    frame['issue']=[r.b.ISSUES[r.b.ISSUES<=t][-1] for t in times]
    frame['target']='up_magnitude_log_bps'
    forecasts=r.candidates(frame)
    assert (forecasts['residual_30d_weight_0.5'][:384]==100).all()
    assert forecasts['residual_30d_weight_0.5'][-1]==110.
    cut=times[8000]
    changed=frame.copy()
    changed.loc[changed.decision_time+pd.Timedelta(hours=72)>=cut,'actual']=1e6
    perturbed=r.candidates(changed)
    for key in forecasts:
        np.testing.assert_array_equal(forecasts[key][times<=cut],perturbed[key][times<=cut])
    assert r.select(frame,forecasts,r.b.ISSUES[0])[0]=='unchanged'
    assert r.select(frame,forecasts,r.b.ISSUES[2])[0]!='unchanged'
    changed=frame.copy()
    changed.loc[changed.decision_time+pd.Timedelta(hours=72)>=r.b.ISSUES[2],'actual']=1e6
    assert r.select(frame,forecasts,r.b.ISSUES[2])==r.select(changed,r.candidates(changed),r.b.ISSUES[2])
    zero=frame.copy();zero.actual=zero.best_model
    assert r.select(zero,r.candidates(zero),r.b.ISSUES[2])[0]=='unchanged'
    down=frame.copy();down.target='down_magnitude_log_bps';down.actual=80.
    pred,decisions=r.run(pd.concat([frame,down]))
    assert len(pred)==2*len(frame) and len(decisions)==8
    for _,part in pred.groupby('target'):
        assert np.mean(abs(part.actual-part.selected))<np.mean(abs(part.actual-part.best_model))
    print('PASS: strict 72h maturity, future isolation, earlier-only selection, zero fallback, independent targets and synthetic bias correction.')


if __name__=='__main__':main()
