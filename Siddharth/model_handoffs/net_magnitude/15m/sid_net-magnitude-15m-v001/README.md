# sid_net-magnitude-15m-v001

Owner: Sid. BTCUSDT, 15 minutes. Frozen quarter starting January 1, 2025 UTC. Tested with Python 3.12.5.

Expected absolute 15-minute simple return in basis points: 10000 * abs(price(t+16min)/price(t+1min)-1), using the archived BTC minute reference price.

## Run the packaged example

Open PowerShell in this folder after extracting the entire bundle. Install the pinned requirements into your Python 3.12.5 environment, then run:

```powershell
python -m pip install -r requirements.txt
python predict.py --input example_input.json --output actual_output.json --artifact-dir "../../../../model_files/sid_net-magnitude-15m-v001"
Get-Content actual_output.json
```

Compare the output with expected_output.json. The synthetic example produces `net_magnitude_bps = 13.6226326407`. Match numeric outputs with absolute and relative tolerance of 1e-8. After uploading, download the manifest-listed files to a local artifact folder and pass that folder to --artifact-dir.

## Inputs

Provide one UTC half-hour feature row. Supply every feature named in INPUT_COLUMNS in predict.py. The frozen estimators select their inputs in the order listed below. example_input.json illustrates the complete interface with artificial feature values. The caller supplies engineered features at the stated origin.

Use BTC/ETH minute paths ending strictly before issuance, one-bar-lagged slow market inputs and only released historical same-slot labels. The feature calculations below document the exact transforms, lags and scale fallbacks.

Ensembling uses 75% histogram boosting and 25% deep MLP, followed by the fitted 0.97 multiplier. Each component output is floored at 0.03 before conversion from volatility units to bps. Supply the positive adjusted volatility scale as log_scale = log(1 + scale). The other inputs allow null for missing data and use the fitted pipeline imputers.

## Evaluation and provenance

evaluation.json includes the matched walk-forward metrics, quarterly breakdown and a separate check of the final checkpoint. manifest.json records the training cutoff, data-catalog and artifact hashes, and the available source provenance. The original training commit has not been established. SHA-256 checks run before loading fitted files. Load trusted artifacts only.

- Quarterly walk-forward evaluation reuses a historical calendar examined in model selection. Evaluation ends 2025-01-23 23:30 UTC. Current market performance is unverified.
- Primary comparison covers successive historical fitted checkpoints. The included frozen January 2025 checkpoint is scored separately in evaluation.json.
- Integration requires the upstream causal feature pipeline and data availability checks. No execution, live feed or quarterly retraining is included.
- Individual move sizes are noisy. In the matched evaluation, 5822 of 23088 were within 25% of actual size.

## Ordered inputs for each fitted component

### boost

```text
BTC_abs_1m_ew15
BTC_abs_1m_ew60
BTC_abs_1m_ew360
BTC_abs_1m_ew1440
BTC_abs_5m_ew15
BTC_abs_5m_ew60
BTC_abs_5m_ew360
BTC_abs_5m_ew1440
BTC_abs_15m_ew15
BTC_abs_15m_ew60
BTC_abs_15m_ew360
BTC_abs_15m_ew1440
BTC_range_15
BTC_range_60
BTC_range_240
ETH_abs_1m_ew15
ETH_abs_1m_ew60
ETH_abs_1m_ew360
ETH_abs_1m_ew1440
ETH_abs_5m_ew15
ETH_abs_5m_ew60
ETH_abs_5m_ew360
ETH_abs_5m_ew1440
ETH_abs_15m_ew15
ETH_abs_15m_ew60
ETH_abs_15m_ew360
ETH_abs_15m_ew1440
ETH_range_15
ETH_range_60
ETH_range_240
log_scale
weekend
ny_slot
ny_weekday
ny_sin1
ny_cos1
ny_sin2
ny_cos2
ny_slot_ratio
utc_slot
utc_weekday
utc_sin1
utc_cos1
utc_sin2
utc_cos2
utc_slot_ratio
minute_trend_15
minute_trend_60
up_energy_60
short_risk_ratio
day_risk_ratio
minute_risk_5
minute_abs_5
minute_trend_5
up_energy_5
minute_risk_15
minute_abs_15
up_energy_15
minute_risk_60
minute_abs_60
minute_risk_240
minute_abs_240
minute_trend_240
up_energy_240
minute_risk_1440
minute_abs_1440
minute_risk_10080
minute_abs_10080
utc_hour
near_utc_0h
near_utc_8h
near_utc_13h
near_utc_14h
near_utc_16h
near_utc_20h
vol_change_5
abs_to_rms_5
vol_change_15
abs_to_rms_15
vol_change_60
abs_to_rms_60
vol_change_240
abs_to_rms_240
same_slot_risk_14d
slot_relative_risk
f4z__flow__flow_activity_percentile__4h
f4z__flow__flow_activity_percentile__1d
f4z__flow__flow_pressure_percentile__4h
f4z__flow__flow_pressure_percentile__1d
f4z__flow__flow_cvd_slope_percentile__4h
f4z__flow__flow_cvd_slope_percentile__1d
f4z__price__price_displacement_percentile__4h
f4z__price__price_displacement_percentile__1d
f4z__price__price_excursion_range_percentile__4h
f4z__price__price_excursion_range_percentile__1d
f4z__volatility__realized_variance_level_percentile__4h
f4z__volatility__realized_variance_level_percentile__1d
risk_2h
trend_over_risk_2h
upside_variance_fraction_2h
risk_8h
trend_over_risk_8h
upside_variance_fraction_8h
risk_16h
trend_over_risk_16h
upside_variance_fraction_16h
risk_48h
trend_over_risk_48h
upside_variance_fraction_48h
risk_168h
trend_over_risk_168h
upside_variance_fraction_168h
risk_672h
trend_over_risk_672h
upside_variance_fraction_672h
imbalance_4h
imbalance_change_4h
flow_price_4h
imbalance_1d
imbalance_change_1d
flow_price_1d
funding_normalized_rate_per_8h
spread_funding_normalized_rate_per_8h
mark_basis_bps
spread_mark_basis_bps
oi_quote_oi_value_usdt_equiv
spread_oi_quote_oi_value_usdt_equiv
hour_sin
hour_cos
weekday_sin
weekday_cos
```

### mlp_deep

```text
BTC_abs_1m_ew15
BTC_abs_1m_ew60
BTC_abs_1m_ew360
BTC_abs_1m_ew1440
BTC_abs_5m_ew15
BTC_abs_5m_ew60
BTC_abs_5m_ew360
BTC_abs_5m_ew1440
BTC_abs_15m_ew15
BTC_abs_15m_ew60
BTC_abs_15m_ew360
BTC_abs_15m_ew1440
BTC_range_15
BTC_range_60
BTC_range_240
ETH_abs_1m_ew15
ETH_abs_1m_ew60
ETH_abs_1m_ew360
ETH_abs_1m_ew1440
ETH_abs_5m_ew15
ETH_abs_5m_ew60
ETH_abs_5m_ew360
ETH_abs_5m_ew1440
ETH_abs_15m_ew15
ETH_abs_15m_ew60
ETH_abs_15m_ew360
ETH_abs_15m_ew1440
ETH_range_15
ETH_range_60
ETH_range_240
log_scale
weekend
ny_slot
ny_weekday
ny_sin1
ny_cos1
ny_sin2
ny_cos2
ny_slot_ratio
utc_slot
utc_weekday
utc_sin1
utc_cos1
utc_sin2
utc_cos2
utc_slot_ratio
minute_trend_15
minute_trend_60
up_energy_60
short_risk_ratio
day_risk_ratio
```

## Feature calculations

The predictor accepts the feature row produced by these calculations. The excerpts document the archived research pipeline. File reads refer to the research workspace. Use equivalent guarded feed inputs in the integrator feature service.


## fit_15m_market_conditional_means.py / minute_features

```python
def minute_features(z, db, index):
    prefix = 'HANDOFF_4_ML_RESEARCH_V1/'
    price_rows = db.read_parquet(BytesIO(z.read(prefix+'data/prices/PRICE_1m_BTCUSDT.parquet')))
    px = price_rows.select("""window_end, CASE WHEN source_structurally_valid AND
        f066__is_valid AND f066__value > 0 AND available_after <= window_end
        AND decision_time <= window_end THEN f066__value END AS px""").df()
    price = np.log(px.set_index('window_end', verify_integrity=True).px.sort_index().asfreq('min'))
    # A completed minute must be strictly earlier than forecast issuance.
    ret = price.diff().shift(1)*10000
    for change in json.loads(z.read(prefix+'catalog/BINDING_TRANSITIONS.json')):
        cut = pd.Timestamp(change['date'], tz='UTC')
        ret.loc[cut:cut+pd.Timedelta(minutes=2)] = np.nan
    x = pd.DataFrame(index=index)
    for n in [5,15,60,240,1440,10080]:
        minimum = max(4, int(.85*n))
        rolling = ret.rolling(n,min_periods=minimum)
        energy = ret.pow(2).rolling(n,min_periods=minimum).mean().reindex(index)
        risk = np.sqrt(15*energy)
        x[f'minute_risk_{n}'] = risk
        x[f'minute_abs_{n}'] = ret.abs().rolling(n,min_periods=minimum).mean().reindex(index)
        if n <= 240:
            x[f'minute_trend_{n}'] = rolling.sum().reindex(index)/(1+np.sqrt(n*energy))
            x[f'up_energy_{n}'] = (ret.clip(lower=0).pow(2).rolling(n,min_periods=minimum).mean().reindex(index)
                                     /energy.replace(0,np.nan))
    x['short_risk_ratio'] = np.log((1+x.minute_risk_15)/(1+x.minute_risk_240))
    x['day_risk_ratio'] = np.log((1+x.minute_risk_240)/(1+x.minute_risk_10080))
    x['utc_hour'] = index.hour + index.minute/60
    x['utc_weekday'] = index.dayofweek
    for hour in [0,8,13,14,16,20]:
        x[f'near_utc_{hour}h'] = np.cos(2*np.pi*(x.utc_hour-hour)/24).clip(lower=0)**12
    scale = (.15*x.minute_risk_15 + .35*x.minute_risk_60 + .30*x.minute_risk_240
             + .20*x.minute_risk_1440).clip(lower=1)
    return x, scale
```

## forecast_16h_market_band.py / market_features

```python
def market_features(db):
    market = db.read_parquet(str(ROOT / 'tmp/conditional_16h_market_inputs.parquet')).df()
    market.index = pd.to_datetime(market.pop('__index_level_0__'), utc=True)
    market = market.filter(regex='^(f4__|f4z__|f1__f066__|f2__|msv1__)')
    duplicates = market[market.index.duplicated(keep=False)]
    assert not duplicates.groupby(level=0).nunique().gt(1).any().any()
    market = market.groupby(level=0).first().asfreq('30min')
    features = market.filter(regex='^f4z__').copy()
    price = np.log(market.f1__f066__value.where(market.f1__f066__value > 0))
    change = 10000 * price.diff()
    for h in [2, 8, 16, 48, 168, 672]:
        rolling = change.rolling(2 * h, min_periods=max(2, int(1.5 * h)))
        risk = (32 * change.pow(2).rolling(2 * h, min_periods=max(2, int(1.5 * h))).mean()).pow(.5)
        features[f'risk_{h}h'] = risk
        features[f'trend_over_risk_{h}h'] = rolling.sum() / risk.replace(0, np.nan)
        features[f'upside_variance_fraction_{h}h'] = change.clip(lower=0).pow(2).rolling(
            2 * h, min_periods=max(2, int(1.5 * h))).mean() * 32 / risk.pow(2).replace(0, np.nan)
    for window in ['4h', '1d']:
        buy = market[f'f2__m011__{window}']
        sell = market[f'f2__m012__{window}']
        imbalance = (buy - sell) / (buy + sell).replace(0, np.nan)
        features[f'imbalance_{window}'] = imbalance
        features[f'imbalance_change_{window}'] = imbalance.diff(8)
        features[f'flow_price_{window}'] = imbalance * features.trend_over_risk_8h
    for suffix in ['funding_normalized_rate_per_8h', 'mark_basis_bps', 'oi_quote_oi_value_usdt_equiv']:
        venues = market.filter(regex=f'^msv1__.*{suffix}$')
        if suffix.startswith('oi_'):
            venues = np.log(venues.where(venues > 0)).diff(8)
        valid = venues.count(axis=1) >= 2
        features[suffix] = venues.median(axis=1).where(valid)
        features[f'spread_{suffix}'] = (venues.max(axis=1) - venues.min(axis=1)).where(valid)
    # Count the fraction of the future horizon on a weekend using only the known clock.
    offsets = range(16)
    features['weekend_fraction'] = np.mean([
        (features.index + pd.Timedelta(hours=i)).dayofweek >= 5 for i in offsets], axis=0)
    for name, angle in [('hour', 2*np.pi*(features.index.hour + features.index.minute/60)/24),
                        ('weekday', 2*np.pi*features.index.dayofweek/7)]:
        features[f'{name}_sin'] = np.sin(angle)
        features[f'{name}_cos'] = np.cos(angle)
    scale = features[['risk_48h', 'risk_168h', 'risk_672h']].median(axis=1).clip(lower=1)
    return features.replace([np.inf, -np.inf], np.nan), scale
```

## search_15m_magnitude_models.py / inputs

```python
def inputs():
    with ZipFile(PRIOR/'Datasets/HANDOFF_4_ML_RESEARCH_V1_MINUTE.zip') as z,duckdb.connect() as db:
        db.execute("SET TimeZone='UTC'")
        frames=[]
        for n in z.namelist():
            if '/data/y_actual/horizon_minutes=15/symbol=BTCUSDT/' in n and n.endswith('.parquet'):
                frames.append(db.read_parquet(BytesIO(z.read(n))).filter("""label_status='VALID'
                    AND epoch_us(decision_time)%1800000000 IN (0,60000000)
                    AND target_end=decision_time+interval '15 minutes' AND target_available_at>=target_end""")
                    .select('decision_time,target_end,target_available_at,signed_return_simple_bps,signed_return_log_bps').df())
        lab=pd.concat(frames).set_index('decision_time',verify_integrity=True).sort_index()
        for transition in json.loads(z.read('HANDOFF_4_ML_RESEARCH_V1/catalog/BINDING_TRANSITIONS.json')):
            t=pd.Timestamp(transition['date'],tz='UTC')
            lab=lab.loc[~((lab.index<t)&(lab.target_end>=t))]
        native=lab[lab.index.minute.isin([0,30])]
        shifted=lab[lab.index.minute.isin([1,31])].copy()
        shifted.index-=pd.Timedelta(minutes=1)
        idx=pd.date_range(min(native.index.min(),shifted.index.min()),max(native.index.max(),shifted.index.max()),freq='30min')
        fast,scale=minute_features(z,db,idx)
        market,_=market_features(db)
        market=market.shift(1).reindex(idx).drop(columns=['weekend_fraction'])
    basic=['minute_risk_15','minute_risk_60','minute_risk_240','minute_risk_1440','minute_risk_10080','utc_hour','utc_weekday']
    new=pd.DataFrame(index=idx)
    for n in [5,15,60,240]:
        risk=fast[f'minute_risk_{n}']
        new[f'vol_change_{n}']=np.log1p(risk)-np.log1p(risk.shift(1))
        new[f'abs_to_rms_{n}']=fast[f'minute_abs_{n}']*np.sqrt(15)/(risk+.05)
    # Known prior days' same half-hour activity preserves intraday seasonality.
    same_hour=pd.concat([fast.minute_risk_15.shift(48*k) for k in range(1,15)],axis=1)
    new['same_slot_risk_14d']=same_hour.median(axis=1)
    new['slot_relative_risk']=np.log1p(fast.minute_risk_15)-np.log1p(new.same_slot_risk_14d)
    x=pd.concat([fast,new,market],axis=1).replace([np.inf,-np.inf],np.nan)
    packs={'core':basic,'minute_path':list(fast),'engineered':list(fast)+list(new),
           'engineered_market':list(x)}
    targets={'Week2_simple_bps':(shifted.signed_return_simple_bps.abs(),shifted.target_available_at),
             'Week3_log_bps':(native.signed_return_log_bps.abs(),native.target_available_at)}
    return x,scale,packs,targets
```

## improve_15m_magnitude_blend.py / asset_features

```python
def asset_features(archive, db, symbol, index, transitions):
    name = f'HANDOFF_4_ML_RESEARCH_V1/data/prices/PRICE_1m_{symbol}USDT.parquet'
    prices = db.read_parquet(BytesIO(archive.read(name))).select("""
        window_end, CASE WHEN source_structurally_valid AND f066__is_valid
        AND f066__value > 0 AND available_after <= window_end
        AND decision_time <= window_end THEN f066__value END AS price
    """).df().set_index('window_end', verify_integrity=True).sort_index().asfreq('min')
    prior = np.log(prices.price).shift(1) * 10000
    for transition in transitions:
        cut = pd.Timestamp(transition['date'], tz='UTC')
        prior.loc[cut:cut + pd.Timedelta(minutes=2)] = np.nan
    features = {}
    for step in (1, 5, 15):
        move = prior.diff(step).abs()
        for minutes in (15, 60, 360, 1440):
            level = move.ewm(halflife=minutes, min_periods=minutes//2).mean().reindex(index)
            features[f'{symbol}_abs_{step}m_ew{minutes}'] = np.log1p(level * np.sqrt(15 / step))
    for minutes in (15, 60, 240):
        roll = prior.rolling(minutes, min_periods=int(.9 * minutes))
        features[f'{symbol}_range_{minutes}'] = np.log1p((roll.max()-roll.min()).reindex(index))
    return pd.DataFrame(features, index=index)
```

## improve_15m_magnitude_blend.py / prepare

```python
def prepare():
    x, old_scale, _, targets = inputs()
    y, release = (v.reindex(x.index) for v in targets['Week2_simple_bps'])
    with ZipFile(PRIOR/'Datasets/HANDOFF_4_ML_RESEARCH_V1_MINUTE.zip') as z, duckdb.connect() as db:
        db.execute("SET TimeZone='UTC'")
        transitions = json.loads(z.read('HANDOFF_4_ML_RESEARCH_V1/catalog/BINDING_TRANSITIONS.json'))
        paths = pd.concat([asset_features(z, db, symbol, x.index, transitions)
                           for symbol in ('BTC', 'ETH')], axis=1)
    level = np.expm1(paths[['BTC_abs_1m_ew15','BTC_abs_5m_ew60','BTC_abs_15m_ew360',
                            'BTC_abs_15m_ew1440']]).mean(axis=1)
    scale = (.75*level + .25*old_scale).clip(lower=1).fillna(old_scale).fillna(15)
    # Each same-slot observation is at least a day old. Exclude late releases
    # before shifting, including rows near archive gaps and other exceptions.
    ratio = (y / scale).where(release < x.index + pd.Timedelta(hours=12)).clip(0, 5)
    calendar = pd.DataFrame(index=x.index)
    calendar['weekend'] = (x.index.dayofweek >= 5).astype(float)
    for name, zone in [('ny','America/New_York'), ('utc','UTC')]:
        clock = x.index.tz_convert(zone)
        slot = clock.hour*2 + clock.minute//30
        business = (clock.dayofweek < 5).astype(int)
        calendar[name+'_slot'] = slot
        calendar[name+'_weekday'] = clock.dayofweek
        for cycle in (1, 2):
            angle = slot*cycle*2*np.pi/48
            calendar[f'{name}_sin{cycle}'] = np.sin(angle)
            calendar[f'{name}_cos{cycle}'] = np.cos(angle)
        # NY fall-back repeats two local slots on the same day: using UTC slot
        # for weekends avoids a same-day duplicate in the historical statistic.
        season_key = np.where(business == 1, slot, 48 + x.index.hour*2 + x.index.minute//30)
        seasonal = ratio.groupby(season_key).transform(
            lambda series: series.shift().ewm(halflife=35, min_periods=8).mean())
        calendar[name+'_slot_ratio'] = seasonal
    adjustment = calendar[['ny_slot_ratio','utc_slot_ratio']].mean(axis=1).fillna(1)
    scale *= (.4 + .6*adjustment.clip(.35,2.5))
    paths['log_scale'] = np.log1p(scale)
    compact = pd.concat([paths,calendar,x[['minute_trend_15','minute_trend_60','up_energy_60',
                                             'short_risk_ratio','day_risk_ratio']]], axis=1)
    # Neural net inputs use log levels and relative measures to control the
    # influence of extreme raw volatility and price ranges.
    all_features = pd.concat([compact, x.drop(columns=compact.columns, errors='ignore')], axis=1)
    return all_features, list(compact), scale, y, release
```
