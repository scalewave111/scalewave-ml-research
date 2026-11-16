"""Fast causal residual calibration of saved best UP neural and DOWN HGB forecasts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import discover_excursion_features_72h as b

REFERENCE = b.HERE/'experiments/focused_hgb_both_72h_v1/20261114T121742176686Z'
NEURAL = b.HERE/'experiments/excursion_neural_72h_v1/20261113T123304519186Z'


def read_verified(path):
    done = json.loads((path/'COMPLETE.json').read_text())
    b.check(done['status']=='COMPLETE' and b.sha(path/'predictions.csv')==done['predictions_sha256'], 'Reference hash')
    frame = pd.read_csv(path/'predictions.csv')
    for c in ['decision_time','issue']:
        frame[c] = pd.to_datetime(frame[c],utc=True)
    b.check(not frame.duplicated(['target','decision_time']).any(), 'Duplicate reference keys')
    return frame


def load():
    data, neural = read_verified(REFERENCE), read_verified(NEURAL)
    keys = ['decision_time','issue','target']
    merged = data.merge(neural[keys+['actual','selected']].rename(columns={'actual':'neural_actual','selected':'neural'}),
                        on=keys, how='outer',validate='one_to_one',indicator=True)
    b.check(merged['_merge'].eq('both').all(), 'Reference alignment')
    np.testing.assert_allclose(merged.actual, merged.neural_actual, rtol=0,atol=1e-10)
    np.testing.assert_allclose(merged.previous_neural, merged.neural, rtol=0,atol=1e-10)
    merged['best_model'] = np.where(merged.target.eq('up_magnitude_log_bps'),merged.neural,merged.candidate)
    return merged


def candidates(frame):
    """Outcome from origin s enters history only at t strictly greater than s+72h."""
    times = pd.DatetimeIndex(frame.decision_time)
    residual = pd.Series((frame.actual-frame.best_model).to_numpy(),index=times+pd.Timedelta(hours=72))
    result = {'unchanged':frame.best_model.to_numpy()}
    for days in [30,60,120]:
        # Union inserts each query timestamp; closed=left excludes outcomes maturing exactly now.
        union = residual.index.union(times).sort_values()
        values = residual.reindex(union).rolling(f'{days}D',closed='left',min_periods=240).median().reindex(times).fillna(0).to_numpy()
        limit = .25*frame.training_median.to_numpy()
        values = np.clip(values,-limit,limit)
        for weight in [.1,.25,.5]:
            result[f'residual_{days}d_weight_{weight}'] = np.maximum(0,result['unchanged']+weight*values)
    return result


def select(frame, forecasts, issue):
    # Last 90 days of fully matured forecast outcomes, split into three 30-day blocks.
    end = issue-pd.Timedelta(hours=72)
    blocks = [(frame.decision_time>=end-pd.Timedelta(days=90-30*j)) &
              (frame.decision_time<end-pd.Timedelta(days=60-30*j)) for j in range(3)]
    if any(mask.sum()<1000 for mask in blocks):
        return 'unchanged', []
    truth = frame.actual.to_numpy()
    baseline = np.array([np.mean(np.abs(truth[m]-forecasts['unchanged'][m])) for m in blocks])
    counts = [m.sum() for m in blocks]
    best, score, rows = 'unchanged', np.average(baseline,weights=counts), []
    for name, pred in forecasts.items():
        errors = np.array([np.mean(np.abs(truth[m]-pred[m])) for m in blocks])
        pooled = np.average(errors,weights=counts)
        eligible = bool(pooled < np.average(baseline,weights=counts)*.999 and
                        (errors<=baseline).sum()>=2 and (errors<=baseline*1.01).all())
        rows.append(dict(candidate=name,mae=float(pooled),window_mae=errors.tolist(),eligible=eligible))
        if eligible and pooled<score:
            best,score = name,pooled
    return best, rows


def run(data):
    frames, decisions = [], []
    for target, part in data.groupby('target'):
        part = part.sort_values('decision_time').reset_index(drop=True)
        forecasts = candidates(part)
        result = forecasts['unchanged'].copy()
        for issue in sorted(part.issue.unique()):
            choice, trials = select(part,forecasts,issue)
            mask = part.issue.eq(issue).to_numpy()
            result[mask] = forecasts[choice][mask]
            decisions.append(dict(target=target,issue=str(issue),choice=choice,trials=trials))
        keep = part[['decision_time','issue','target','actual','best_model','training_median']].copy()
        keep['selected'] = result
        frames.append(keep)
    return pd.concat(frames,ignore_index=True), decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-inputs',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=b.HERE/'experiments/best_models_calibration_72h_v1')
    args = parser.parse_args()
    data = load()
    print(f'Checked {len(data)} target/origin pairs and both source hashes.',flush=True)
    if args.check_inputs:
        return
    out = args.output_dir/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=False)
    pred, decisions = run(data)
    replay, replay_decisions = run(data)
    pd.testing.assert_frame_equal(pred,replay)
    b.check(decisions==replay_decisions,'Calibration replay')
    pred['period'] = pred.decision_time.dt.tz_localize(None).dt.to_period('Q').astype(str)
    pred.loc[pred.decision_time>=pd.Timestamp('2025-01-01',tz='UTC'),'period']='2025-01'
    pred.to_csv(out/'predictions.csv',index=False)
    rows=[]
    for target, frame in pred.groupby('target'):
        for period, part in [('ALL',frame)]+list(frame.groupby('period')):
            for name in ['selected','best_model','training_median']:
                rows.append(dict(target=target,period=period,model=name,**b.metric(part.actual.to_numpy(),part[name].to_numpy())))
    scores=pd.DataFrame(rows)
    scores.to_csv(out/'metrics.csv',index=False)
    b.dump(out/'selections.json',decisions)
    paths=[Path(__file__),Path(b.__file__)]
    b.dump(out/'CONTRACT.json',dict(arguments=vars(args),horizon_hours=72,
        bases=dict(up='saved neural selected',down='saved focused HGB candidate'),
        lookbacks_days=[30,60,120],weights=[.1,.25,.5],clip='25% of quarterly training median',
        minimum_residual_observations=240,validation_days=90,validation_blocks=3,minimum_block_rows=1000,
        gate='At least .1% MAE improvement vs best model, 2/3 block wins or ties, no block >1% worse; else unchanged',
        availability='Strict origin+72h<decision_time for residuals and validation; rolling updates use matured outcomes during evaluation',
        limitations=['Historical adaptive research; bases chosen using exposed evaluation',
                     'DOWN base remains diagnostic candidate that failed original consistency gates',
                     'Not a frozen batch forecast: requires live matured labels to update correction',
                     'No underlying model retraining; no guaranteed gain; minimum-history fallback'],
        input_hashes={str(p/n):b.sha(p/n) for p in [REFERENCE,NEURAL] for n in ['COMPLETE.json','predictions.csv']},
        source_hashes={str(p):b.sha(p) for p in paths}))
    for p in paths:
        (out/p.name).write_bytes(p.read_bytes())
    summary='Calibration on saved best models; exposed historical research.\n'+scores[scores.period.eq('ALL')].to_string(index=False)
    (out/'SUMMARY.txt').write_text(summary)
    b.dump(out/'COMPLETE.json',dict(status='COMPLETE',rows=len(pred),replay_passed=True,predictions_sha256=b.sha(out/'predictions.csv')))
    print(summary)
    print('Saved:',out)


if __name__=='__main__':
    main()
