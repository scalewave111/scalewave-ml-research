"""Fresh 72h candidate universe; no original F4Z/NEWPRICE predictors."""
import numpy as np
import pandas as pd
import long_horizon_features as shared


def build(r, origins, horizon=72):
    if horizon != 72:
        raise ValueError('This feature design is for 72h only')
    base, dictionary = shared.build(r, origins, horizon)
    hour = r.resample('1h', closed='left', label='right').sum(min_count=12)
    hour = hour.where(r.resample('1h', closed='left', label='right').count().eq(12))
    values, definitions = {}, []
    for h in [24, 72, 168, 720]:
        roll = hour.rolling(h, min_periods=h)
        valid = roll.count().eq(h)
        scale = np.sqrt(hour.pow(2).rolling(h, min_periods=h).mean()).clip(lower=1e-8)
        # h-1 adjacent pairs wholly inside the same h-hour window.
        pairs = (hour * hour.shift(1)).rolling(h-1, min_periods=h-1).mean()
        fields = {
            'positive_tail': (roll.quantile(.9)/scale, 'P90(hourly return) / RMS(hourly return)'),
            'negative_tail': (-roll.quantile(.1)/scale, '-P10(hourly return) / RMS(hourly return)'),
            'adjacent_movement': (pairs/scale.pow(2), 'mean of h-1 adjacent return products / mean squared return; uncentered, not correlation'),
            'activity_dispersion': (hour.abs().rolling(h,min_periods=h).std(ddof=0)/hour.abs().rolling(h,min_periods=h).mean().clip(lower=1e-8), 'population std(abs hourly return) / mean(abs hourly return)'),
        }
        for name, (value, formula) in fields.items():
            key = f'FRESH72__{name}__{h}h'
            values[key] = value.where(valid)
            definitions.append(dict(feature=key,family='hourly_distribution',measurement=name,lookback=f'{h}h',formula=formula+'; denominator floor 1e-8'))
    extra = pd.DataFrame(values).replace([np.inf,-np.inf],np.nan).reindex(origins,method='ffill',tolerance=pd.Timedelta(minutes=59,seconds=59))
    return pd.concat([base,extra],axis=1),pd.concat([dictionary,pd.DataFrame(definitions)],ignore_index=True)
