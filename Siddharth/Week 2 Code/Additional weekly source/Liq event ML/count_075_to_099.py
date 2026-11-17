"""Descriptive threshold episodes on prepared half-hour BTCUSDT observations."""
from pathlib import Path
import sys
import json
import duckdb
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
WEEK1 = OUT.parent.parent / 'Week 1'
sys.path.insert(0, str(WEEK1 / 'Code'))
import canonical_return_features as canonical

FEATURE = 'F4Z__liquidation__liq_activity_severity_active_severity_percentile__4h__value'


def main():
    frames = []
    with duckdb.connect() as c:
        for split in ['train', 'development']:
            path = WEEK1 / f'Dataset/TEAM_48H_PLAYGROUND_V1/TEAM_48H_PLAYGROUND_V1/data/X_{split}.parquet'
            f = c.execute(f'SELECT decision_time,"{FEATURE}" AS value FROM read_parquet(?)', [str(path)]).fetchdf()
            f['split'] = split
            frames.append(f)
    f = pd.concat(frames).sort_values('decision_time')
    f['decision_time'] = pd.to_datetime(f.decision_time, utc=True)
    assert f.decision_time.is_unique
    episodes, active = [], None
    previous_time, previous_value = None, np.nan
    direct, censored_starts = 0, 0
    for row in f.itertuples(index=False):
        t, value, split = row.decision_time, row.value, row.split
        consecutive = previous_time is not None and t-previous_time == pd.Timedelta(minutes=30)
        if not consecutive or not np.isfinite(value) or not np.isfinite(previous_value):
            if active is not None:
                active.update(status='censored_gap', end_time=previous_time)
                episodes.append(active); active = None
            if np.isfinite(value) and value > .75:
                censored_starts += 1
        elif active is not None:
            if value >= .99:
                active.update(status='reached_099', end_time=t, end_value=value)
                episodes.append(active); active = None
            elif value <= .75:
                active.update(status='fell_back', end_time=t, end_value=value)
                episodes.append(active); active = None
        elif previous_value <= .75 and value > .75:
            if value >= .99:
                direct += 1
            else:
                active = dict(start_time=t, start_value=value, split=split)
        previous_time, previous_value = t, value
    if active is not None:
        active.update(status='censored_end', end_time=previous_time)
        episodes.append(active)
    result = pd.DataFrame(episodes)
    returns = canonical.load_returns(WEEK1 / 'Dataset/canonical_returns_5m.parquet')
    for i, row in result[result.status.eq('reached_099')].iterrows():
        window = returns[(returns.index > row.start_time) & (returns.index <= row.end_time)]
        expected = int((row.end_time-row.start_time)/pd.Timedelta(minutes=5))
        assert len(window) == expected and expected > 0
        result.loc[i, 'minutes_to_099'] = (row.end_time-row.start_time).total_seconds()/60
        result.loc[i, 'price_return_available'] = bool(window.notna().all())
        if window.notna().all():
            result.loc[i, 'signed_return_bps'] = 10000*np.expm1(window.sum()/10000)
    hits = result[result.status.eq('reached_099')]
    summary = dict(feature=FEATURE, observed_rows=len(f), sampling='30-minute prepared observations, gaps break episodes',
        definition='Observed crossing from <=0.75 into (0.75,0.99), then first >=0.99 before returning <=0.75',
        start_episodes=len(result), reached_099=len(hits), fell_back=int(result.status.eq('fell_back').sum()),
        censored_episodes=int(result.status.str.startswith('censored').sum()), direct_jumps_to_099_excluded=direct,
        unobserved_entry_above_075_after_gap=censored_starts,
        priced_hits=int(hits.signed_return_bps.notna().sum()),
        mean_signed_return_bps=float(hits.signed_return_bps.mean()),
        median_signed_return_bps=float(hits.signed_return_bps.median()),
        mean_absolute_return_bps=float(hits.signed_return_bps.abs().mean()),
        mean_minutes_to_099=float(hits.minutes_to_099.mean()))
    result.to_csv(OUT / 'episodes_075_to_099.csv', index=False)
    (OUT / 'episodes_075_to_099_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
