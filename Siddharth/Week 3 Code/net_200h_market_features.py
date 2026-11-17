"""Strictly historical market-state features from masked half-hour snapshots."""
import numpy as np
import pandas as pd

SOURCES={
 'funding_level':'F4Z__funding__funding_level_percentile__5m__value',
 'funding_change':'F4Z__funding__funding_change_percentile__5m__value',
 'basis_level':'F4Z__basis__basis_level_percentile__5m__value',
 'basis_change':'F4Z__basis__basis_change_percentile__5m__value',
 'oi_position':'F4Z__oi__oi_range_position_percentile__5m__value',
 'oi_unwind':'F4Z__oi__leverage_unwind_percentile__5m__value',
 'flow_pressure':'F4Z__flow__flow_pressure_percentile__5m__value',
 'flow_activity':'F4Z__flow__flow_activity_percentile__5m__value',
 'liq_history':'F4Z__liquidation__liq_activity_severity_historical_active_rate__5m__value',
 'peer_liq_history':'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__5m__value',
 'liq_breadth':'F4Z__liquidation__venue_breadth_percentile__5m__value',
}

def load_sources(data_dir,origins):
    """Read original masked X values, not the reduced legacy model inventory."""
    from pathlib import Path
    import duckdb
    paths=[Path(data_dir)/name for name in ['X_train.parquet','X_development.parquet']]
    columns=list(SOURCES.values())
    projection=','.join('"'+c+'"' for c in ['decision_time']+columns)
    with duckdb.connect() as con:
        frames=[con.execute('SELECT '+projection+' FROM read_parquet(?)',[str(path)]).fetchdf() for path in paths]
    frame=pd.concat(frames,ignore_index=True)
    frame['decision_time']=pd.to_datetime(frame.decision_time,utc=True).dt.as_unit('ns')
    frame=frame.set_index('decision_time').sort_index()
    if not frame.index.is_unique:raise ValueError('Duplicate original masked X timestamps')
    missing=origins.difference(frame.index)
    if len(missing):raise ValueError(f'Original masked X missing {len(missing)} required timestamps')
    return frame.reindex(origins).replace([np.inf,-np.inf],np.nan),paths


def build(market,price,origins):
    if not market.index.is_unique or not market.index.is_monotonic_increasing:raise ValueError('Invalid market keys')
    missing=sorted(set(SOURCES.values())-set(market))
    if missing:raise ValueError('Missing required market source columns: '+', '.join(missing))
    grid=pd.date_range(market.index.min(),market.index.max(),freq='30min')
    if not market.index.isin(grid).all():raise ValueError('Market timestamps not on half-hour grid')
    data=market[list(SOURCES.values())].reindex(grid).replace([np.inf,-np.inf],np.nan)
    values={};definitions=[]
    def add(name,v,formula,sources):
        values[name]=v.reindex(origins)
        definitions.append(dict(feature=name,family='market_state',formula=formula,sources=sources))
    means={}
    for name,col in SOURCES.items():
        s=data[col];lag=s.shift(1)
        add(f'STATE__{name}__lag30m',lag,'Snapshot at t-30min; missing grid values remain missing',col)
        for h in [6,24,200,168]:
            roll=s.rolling(f'{h}h',closed='left',min_periods=int(np.ceil(.8*2*h)))
            mean=roll.mean();means[name,h]=mean.reindex(origins)
            high=s.ge(.8).where(s.notna()).rolling(f'{h}h',closed='left',min_periods=int(np.ceil(.8*2*h))).mean()
            for stat,v,formula in [('mean',mean,'mean'),('std',roll.std(ddof=0),'population std'),
                ('deviation',lag-mean,'lag30m minus mean'),('high_share',high,'fraction of observed snapshots >=.8')]:
                add(f'STATE__{name}_{stat}__{h}h',v,f'{formula} of snapshots in [t-{h}h,t); >=80% expected half-hour coverage',col)
    # Interactions are hypotheses; names describe products, not causal events.
    for h in [24,200,168]:
        a=lambda name: means[name,h]
        momentum=price[f'LONG__normalized_momentum__{h}h']
        volatility=price[f'LONG__vol_per_sqrt_hour__{h}h']
        pairs={
            'funding_oi':(a('funding_level')*a('oi_position'),'funding_level mean * oi_position mean'),
            'oi_lowflow':(a('oi_position')*(1-a('flow_pressure')),'oi_position mean * (1-flow_pressure mean)'),
            'unwind_negative_momentum':(a('oi_unwind')*(-momentum).clip(lower=0),'oi_unwind mean * max(-price normalized momentum,0)'),
            'liq_volatility':(a('liq_history')*volatility,'historical liquidation activity-rate mean * hourly price RMS'),
            'basis_funding':((a('basis_level')-.5)*(a('funding_level')-.5),'(basis_level mean-.5)*(funding_level mean-.5)'),
            'flow_momentum':((a('flow_pressure')-.5)*momentum,'(flow_pressure mean-.5)*price normalized momentum'),
        }
        for name,(v,formula) in pairs.items():add(f'INTERACT__{name}__{h}h',v,formula+'; same past-hour window', 'SOURCES and LONG price features')
    fresh=pd.DataFrame(values,index=origins).replace([np.inf,-np.inf],np.nan)
    return fresh,pd.DataFrame(definitions)
