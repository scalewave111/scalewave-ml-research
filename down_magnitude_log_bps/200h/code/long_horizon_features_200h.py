"""Causal candidate features for 200h excursions; no targets used here."""
import numpy as np
import pandas as pd

HOURS = [24, 72, 168, 200, 336, 720, 1440]
MARKET = {
    'basis': 'basis_level_percentile',
    'funding': 'funding_level_percentile',
    'flow': 'flow_pressure_percentile',
    'oi': 'oi_range_position_percentile',
    'liquidation': 'liq_activity_severity_active_severity_percentile',
}


def make_features(r, origins, market=None):
    """5m endpoint returns in log bps; every window excludes the origin.

    Price paths are relative cumulative returns, not exchange intrabar extremes.
    Missing returns invalidate a whole affected price window. Market aggregates
    use the original half-hour X inventory and require >=80% expected coverage.
    """
    if not r.index.is_unique or not r.index.is_monotonic_increasing:
        raise ValueError('Invalid return index')
    if not (np.diff(r.index.asi8) == pd.Timedelta(minutes=5).value).all():
        raise ValueError('Returns must be a regular five-minute grid')
    values, definitions = {}, []
    level = r.fillna(0).cumsum().shift(1)
    def add(name, value, formula, group):
        values[name] = value.reindex(origins)
        definitions.append(dict(feature=name, formula=formula, group=group))
    cache = {}
    for h in HOURS:
        n = h*12
        opts = dict(window=f'{h}h', closed='left', min_periods=n)
        net = r.rolling(**opts).sum()
        rv = np.sqrt(r.pow(2).rolling(**opts).sum())
        absolute = r.abs().rolling(**opts).sum()
        up2 = r.clip(lower=0).pow(2).rolling(**opts).sum()
        down2 = r.clip(upper=0).pow(2).rolling(**opts).sum()
        high = level.rolling(n+1, min_periods=n+1).max()
        low = level.rolling(n+1, min_periods=n+1).min()
        width = high-low
        valid = r.rolling(**opts).count().eq(n)
        stats = {
            'net': (net, 'sum(r)'),
            'rv': (rv, 'sqrt(sum(r^2))'),
            'up_rv': (np.sqrt(up2), 'sqrt(sum(max(r,0)^2))'),
            'down_rv': (np.sqrt(down2), 'sqrt(sum(min(r,0)^2))'),
            'range': (width, 'max(relative log path)-min(relative log path)'),
            'drawdown': (high-level, 'max(relative log path)-last relative log level'),
            'rally': (level-low, 'last relative log level-min(relative log path)'),
            'range_position': ((level-low)/width.clip(lower=1e-8), 'rally/max(range,1e-8)'),
            'trend_efficiency': (net/absolute.clip(lower=1e-8), 'sum(r)/max(sum(abs(r)),1e-8)'),
            'trend_to_vol': (net/rv.clip(lower=1e-8), 'sum(r)/max(rv,1e-8)'),
            'variance_balance': ((up2-down2)/(up2+down2).clip(lower=1e-8), '(up_variance-down_variance)/max(total_variance,1e-8)'),
            'jump_share': (r.pow(2).rolling(**opts).max()/rv.pow(2).clip(lower=1e-8), 'max(r^2)/max(sum(r^2),1e-8)'),
        }
        for key, (v, formula) in stats.items():
            add(f'LONGPRICE__{key}__{h}h', v.where(valid), f'{h}h past window: {formula}', 'price')
        cache[h] = (rv.where(valid), net.where(valid))
    for short, long in [(24,200), (72,200), (168,720), (200,720), (336,1440), (720,1440)]:
        rv_s, net_s = cache[short]
        rv_l, net_l = cache[long]
        add(f'LONGPRICE__vol_ratio__{short}_{long}h',
            (rv_s/np.sqrt(short))/(rv_l/np.sqrt(long)).clip(lower=1e-8),
            f'(rv_{short}/sqrt({short}))/(rv_{long}/sqrt({long}))', 'regime')
        add(f'LONGPRICE__trend_acceleration__{short}_{long}h',
            net_s/short-(net_l-net_s)/(long-short),
            f'net_{short}/{short}-(net_{long}-net_{short})/{long-short} log bps/hour', 'regime')
    if market is not None:
        for family, measure in MARKET.items():
            col = f'F4Z__{family}__{measure}__4h__value'
            if col not in market:
                raise ValueError('Missing market source '+col)
            s = market[col].replace([np.inf, -np.inf], np.nan)
            s = s.reindex(s.index.union(origins).sort_values())
            for h in [168, 720]:
                opts = dict(window=f'{h}h', closed='left', min_periods=int(np.ceil(.8*h*2)))
                for stat, v in [('mean', s.rolling(**opts).mean()),
                                ('std', s.rolling(**opts).std(ddof=0))]:
                    add(f'LONGMARKET__{family}_{stat}__{h}h', v,
                        f'{stat} of prior observed {col}; >=80% expected half-hour rows over {h}h; no filling', 'market')
    return pd.DataFrame(values, index=origins).replace([np.inf,-np.inf], np.nan), definitions
