"""Causal multiday features from canonical endpoint-indexed 5m log-bps returns."""
import numpy as np
import pandas as pd


def build(r, origins, horizon):
    if not r.index.is_unique or not r.index.is_monotonic_increasing:
        raise ValueError('Return timestamps must be unique and sorted')
    if len(r) > 1 and not (np.diff(r.index.asi8) == pd.Timedelta(minutes=5).value).all():
        raise ValueError('Expected uninterrupted 5-minute timestamp grid; gaps must be NaN')
    # A bin labelled t contains endpoints [t-1h,t), so nothing at t is used.
    hour = r.resample('1h', closed='left', label='right').sum(min_count=12)
    count = r.resample('1h', closed='left', label='right').count()
    hour = hour.where(count.eq(12))
    level = hour.fillna(0).cumsum()
    lookbacks = sorted(set([24, max(24, horizon//2), horizon, 2*horizon, 168, 336, 720, 1440]))
    values, dictionary = {}, []
    def add(name, v, h, formula):
        key = f'LONG__{name}__{h}h'
        values[key] = v
        dictionary.append(dict(feature=key, family='long_price', measurement=name,
                               lookback=f'{h}h', formula=formula))
    for h in lookbacks:
        roll = hour.rolling(h, min_periods=h)
        valid = roll.count().eq(h)
        net = roll.sum()
        rv2 = hour.pow(2).rolling(h, min_periods=h).sum()
        rv = np.sqrt(rv2)
        up2 = hour.clip(lower=0).pow(2).rolling(h, min_periods=h).sum()
        absolute = hour.abs().rolling(h, min_periods=h).sum()
        high = level.rolling(h+1, min_periods=h+1).max()
        low = level.rolling(h+1, min_periods=h+1).min()
        width = (high-low).where(valid)
        safe_rv, safe_width = rv.clip(lower=1e-8), width.clip(lower=1e-8)
        fields = {
            'net': (net, 'sum hourly log returns'),
            'vol_per_sqrt_hour': (rv/np.sqrt(h), 'sqrt(sum hourly return squared / hours)'),
            'up_variance_share': (up2/rv2.clip(lower=1e-16), 'sum positive hourly return squared / total squared'),
            'trend_efficiency': (net/absolute.clip(lower=1e-8), 'net / sum absolute hourly returns'),
            'normalized_momentum': (net/safe_rv, 'net / realized volatility'),
            'range': (width, 'maximum minus minimum local cumulative hourly log price'),
            'range_position': ((level-low)/safe_width, '(current minus local low) / range'),
            'drawdown_vol_units': ((high-level)/safe_rv, '(local high minus current) / volatility'),
            'rebound_vol_units': ((level-low)/safe_rv, '(current minus local low) / volatility'),
            'positive_hour_share': (hour.gt(0).where(hour.notna()).rolling(h,min_periods=h).mean(), 'fraction positive hourly returns'),
            'jump_variance_share': (hour.pow(2).rolling(h,min_periods=h).max()/rv2.clip(lower=1e-16), 'largest squared hour / total squared hours'),
            'hourly_skew': (roll.skew(), 'sample skewness of hourly returns'),
        }
        for name, (v, formula) in fields.items():
            add(name, v.where(valid), h, formula)
    for short, long in [(24,horizon), (horizon,2*horizon), (168,720), (336,1440)]:
        a = values[f'LONG__vol_per_sqrt_hour__{short}h']
        z = values[f'LONG__vol_per_sqrt_hour__{long}h']
        add(f'vol_ratio_{short}', a/z.clip(lower=1e-8), long, f'volatility per sqrt hour {short}h / {long}h')
        a = values[f'LONG__net__{short}h']/short
        z = values[f'LONG__net__{long}h']/long
        add(f'drift_change_{short}', a-z, long, f'mean hourly drift {short}h minus {long}h')
    frame = pd.DataFrame(values).replace([np.inf,-np.inf],np.nan)
    # Latest completed hour only, at most 59m59s old; never backfill early rows.
    frame = frame.reindex(origins, method='ffill', tolerance=pd.Timedelta(minutes=59,seconds=59))
    return frame, pd.DataFrame(dictionary)
