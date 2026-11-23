# sid_direction-15m-v001

Owner: Sid. BTCUSDT, 15 minutes. Frozen quarter starting January 1, 2025 UTC. Tested with Python 3.12.5.

Probability of a positive native 15-minute BTCUSDT endpoint return from t to t+15. The reported execution-target comparison evaluates the sign from t+1 to t+16.

## Run the packaged example

Open PowerShell in this folder after extracting the entire bundle. Install the pinned requirements into your Python 3.12.5 environment, then run:

```powershell
python -m pip install -r requirements.txt
python predict.py --input example_input.json --output actual_output.json --artifact-dir "../../../../model_files/sid_direction-15m-v001"
Get-Content actual_output.json
```

Compare the output with expected_output.json. The synthetic example produces `p_up = 0.492407167581`. Match numeric outputs with absolute and relative tolerance of 1e-8. After uploading, download the manifest-listed files to a local artifact folder and pass that folder to --artifact-dir.

## Inputs

Provide one UTC half-hour feature row. Supply every feature named in INPUT_COLUMNS in predict.py. The frozen estimators select their inputs in the order listed below. example_input.json illustrates the complete interface with artificial feature values. The caller supplies engineered features at the stated origin.

Use only origin-available market features and trailing short-window BTC inputs. Apply the validity masks and release checks described below.

Ensembling uses 75% shallow histogram boosting and 25% ridge logistic probabilities. The columns are aligned by name to the per-estimator order before clipping. Null encodes missing data. Boosting handles missing values natively and the ridge pipeline uses its fitted imputer. When an entire component row is missing, its saved prior is used before applying the saved probability transform.

## Evaluation and provenance

evaluation.json includes the matched walk-forward metrics, quarterly breakdown and a separate check of the final checkpoint. manifest.json records the training cutoff, data-catalog and artifact hashes, and the available source provenance. The original training commit has not been established. SHA-256 checks run before loading fitted files. Load trusted artifacts only.

- Quarterly walk-forward evaluation reuses a historical calendar examined in model selection. Evaluation ends 2025-01-23 23:30 UTC. Current market performance is unverified.
- Primary comparison covers successive historical fitted checkpoints. The included frozen January 2025 checkpoint is scored separately in evaluation.json.
- Integration requires the upstream causal feature pipeline and data availability checks. No execution, live feed or quarterly retraining is included.
- Native training sign and one-minute-delayed execution sign can differ.

## Ordered inputs for each fitted component

### boost

```text
carry_agreement
f2_basis_change
f2_excursion_balance
f2_funding_change
f2_liq_concentration
f2_liq_peak_fraction
f2_liq_share
f2_oi_change
f2_oi_price_alignment
f2_range_spike
f2_taker_balance
f2_taker_share
f2_variance_turnover_gap
f4__basis__basis_level__4h
f4__flow__flow_activity__1d
f4__flow__flow_cvd_slope__1d
f4__flow__flow_pressure__1d
f4__flow__flow_pressure__4h
f4__funding__funding_level__4h
f4__liquidation__liq_activity_severity__4h
f4__liquidation__liq_side_pressure__4h
f4__liquidation__venue_fragmentation__4h
f4__oi__leverage_unwind__4h
f4__oi__oi_range_position__4h
f4__price__price_displacement__1d
f4__price__price_displacement__4h
f4__price__price_excursion_range__1d
f4__price__price_move_retention__4h
f4__volatility__realized_variance_level__1d
f4__volatility__realized_variance_level__4h
f4z__flow__flow_activity_percentile__4h
f4z__flow__flow_cvd_slope_percentile__4h
f4z__flow__flow_pressure_percentile__4h
f4z__price__price_displacement_percentile__4h
f4z__price__price_excursion_range_percentile__4h
f4z__volatility__realized_variance_level_percentile__4h
f5__flow_price__flow_price_alignment__1d_leaf_2
f5__flow_price__flow_price_alignment__1d_leaf_3
f5__flow_price__flow_price_alignment__1d_leaf_5
f5__flow_price__flow_price_alignment__1d_leaf_6
f5__flow_price__flow_price_alignment__4h_leaf_2
f5__flow_price__flow_price_alignment__4h_leaf_3
f5__flow_price__flow_price_alignment__4h_leaf_5
f5__flow_price__flow_price_alignment__4h_leaf_6
f5__flow_price__price_response_per_flow__1d_leaf_2
f5__flow_price__price_response_per_flow__1d_leaf_3
f5__flow_price__price_response_per_flow__1d_leaf_5
f5__flow_price__price_response_per_flow__1d_leaf_6
f5__liq_oi__oi_response_per_liq__4h_leaf_2
f5__liq_oi__oi_response_per_liq__4h_leaf_3
f5__liq_oi__oi_response_per_liq__4h_leaf_5
f5__liq_oi__oi_response_per_liq__4h_leaf_6
f5__liq_price__liq_price_alignment__4h_leaf_2
f5__liq_price__liq_price_alignment__4h_leaf_3
f5__liq_price__liq_price_alignment__4h_leaf_5
f5__liq_price__liq_price_alignment__4h_leaf_6
f5__price_vol__price_displacement_to_vol__4h_leaf_2
f5__price_vol__price_displacement_to_vol__4h_leaf_3
f5__price_vol__price_displacement_to_vol__4h_leaf_5
f5__price_vol__price_displacement_to_vol__4h_leaf_6
f5z__flow_price__flow_price_alignment_percentile__4h_leaf_2
f5z__flow_price__flow_price_alignment_percentile__4h_leaf_3
f5z__flow_price__flow_price_alignment_percentile__4h_leaf_5
f5z__flow_price__flow_price_alignment_percentile__4h_leaf_6
f5z__liq_oi__oi_response_per_liq_percentile__4h_leaf_2
f5z__liq_oi__oi_response_per_liq_percentile__4h_leaf_3
f5z__liq_oi__oi_response_per_liq_percentile__4h_leaf_5
f5z__liq_oi__oi_response_per_liq_percentile__4h_leaf_6
f5z__liq_price__liq_price_alignment_percentile__4h_leaf_2
f5z__liq_price__liq_price_alignment_percentile__4h_leaf_3
f5z__liq_price__liq_price_alignment_percentile__4h_leaf_5
f5z__liq_price__liq_price_alignment_percentile__4h_leaf_6
f5z__price_vol__price_displacement_to_vol_percentile__4h_leaf_2
f5z__price_vol__price_displacement_to_vol_percentile__4h_leaf_3
f5z__price_vol__price_displacement_to_vol_percentile__4h_leaf_5
f5z__price_vol__price_displacement_to_vol_percentile__4h_leaf_6
flow_price_agreement
forced_flow_unwind
move_per_range
move_per_volatility
positive_days_7d
pressure_change_72h
share_acceleration
volatility_share
taker_imbalance_15m
imbalance_shift_two_15m_buckets
log_vwap_shift_two_15m_buckets
flow_price_alignment_15m
```

### ridge

```text
f4__flow__flow_activity__1d
f4__flow__flow_cvd_slope__1d
f4__flow__flow_pressure__1d
f4__price__price_displacement__1d
f4__price__price_excursion_range__1d
f4__volatility__realized_variance_level__1d
f4z__flow__flow_activity_percentile__4h
f4z__flow__flow_cvd_slope_percentile__4h
f4z__flow__flow_pressure_percentile__4h
f4z__price__price_displacement_percentile__4h
f4z__price__price_excursion_range_percentile__4h
f4z__volatility__realized_variance_level_percentile__4h
taker_imbalance_15m
imbalance_shift_two_15m_buckets
log_vwap_shift_two_15m_buckets
flow_price_alignment_15m
```

## Feature calculations

The predictor accepts the feature row produced by these calculations. The excerpts document the archived research pipeline. File reads refer to the research workspace. Use equivalent guarded feed inputs in the integrator feature service.

Use the supplied pattern_rules.csv from the private artifact directory when forming the fitted leaf indicators. The F4/F4Z/F2 feature families use the Handoff 4 registry definitions.

- `carry_agreement` ? Product of 4h funding and basis signs; zero is preserved.
- `f2_basis_change` ? 4h basis change in basis points.
- `f2_excursion_balance` ? 4h signed balance of positive and negative VWAP excursions.
- `f2_funding_change` ? 4h genuine funding change (rate per 8h).
- `f2_liq_concentration` ? 4h liquidation concentration across venues (HHI).
- `f2_liq_peak_fraction` ? Largest source-clock liquidation bucket / 4h total.
- `f2_liq_share` ? 4h / 1d published liquidation total.
- `f2_oi_change` ? Change in genuine 4h open interest / starting OI.
- `f2_oi_price_alignment` ? 4h relative OI change times sign of VWAP return.
- `f2_range_spike` ? 4h Binance maximum / mean source-clock true range.
- `f2_taker_balance` ? 4h net taker quote / total taker quote.
- `f2_taker_share` ? 4h / 1d total taker quote (M011 + M012).
- `f2_variance_turnover_gap` ? 4h share of realized variance less 4h share of taker quote.
- `f4__basis__basis_level__4h` ? Native archived market measurement.
- `f4__flow__flow_activity__1d` ? Native archived market measurement.
- `f4__flow__flow_cvd_slope__1d` ? Native archived market measurement.
- `f4__flow__flow_pressure__1d` ? Native archived market measurement.
- `f4__flow__flow_pressure__4h` ? Native archived market measurement.
- `f4__funding__funding_level__4h` ? Native archived market measurement.
- `f4__liquidation__liq_activity_severity__4h` ? Native archived market measurement.
- `f4__liquidation__liq_side_pressure__4h` ? Native archived market measurement.
- `f4__liquidation__venue_fragmentation__4h` ? Native archived market measurement.
- `f4__oi__leverage_unwind__4h` ? Native archived market measurement.
- `f4__oi__oi_range_position__4h` ? Native archived market measurement.
- `f4__price__price_displacement__1d` ? Native archived market measurement.
- `f4__price__price_displacement__4h` ? Native archived market measurement.
- `f4__price__price_excursion_range__1d` ? Native archived market measurement.
- `f4__price__price_move_retention__4h` ? Native archived market measurement.
- `f4__volatility__realized_variance_level__1d` ? Native archived market measurement.
- `f4__volatility__realized_variance_level__4h` ? Native archived market measurement.
- `f4z__flow__flow_activity_percentile__4h` ? Native archived percentile of the corresponding four-hour measure.
- `f4z__flow__flow_cvd_slope_percentile__4h` ? Native archived percentile of the corresponding four-hour measure.
- `f4z__flow__flow_pressure_percentile__4h` ? Native archived percentile of the corresponding four-hour measure.
- `f4z__price__price_displacement_percentile__4h` ? Native archived percentile of the corresponding four-hour measure.
- `f4z__price__price_excursion_range_percentile__4h` ? Native archived percentile of the corresponding four-hour measure.
- `f4z__volatility__realized_variance_level_percentile__4h` ? Native archived percentile of the corresponding four-hour measure.
- `f5__flow_price__flow_price_alignment__1d_leaf_2` ? 1-day flow pressure <= -0.020960563 **and** 1-day flow pressure <= -0.029916405
- `f5__flow_price__flow_price_alignment__1d_leaf_3` ? 1-day flow pressure <= -0.020960563 **and** 1-day flow pressure > -0.029916405
- `f5__flow_price__flow_price_alignment__1d_leaf_5` ? 1-day flow pressure > -0.020960563 **and** 1-day flow pressure <= 0.016709449
- `f5__flow_price__flow_price_alignment__1d_leaf_6` ? 1-day flow pressure > -0.020960563 **and** 1-day flow pressure > 0.016709449
- `f5__flow_price__flow_price_alignment__4h_leaf_2` ? 4-hour flow pressure <= 0.039684495 **and** 4-hour flow pressure <= -0.032701695
- `f5__flow_price__flow_price_alignment__4h_leaf_3` ? 4-hour flow pressure <= 0.039684495 **and** 4-hour flow pressure > -0.032701695
- `f5__flow_price__flow_price_alignment__4h_leaf_5` ? 4-hour flow pressure > 0.039684495 **and** 4-hour flow pressure <= 0.058576462
- `f5__flow_price__flow_price_alignment__4h_leaf_6` ? 4-hour flow pressure > 0.039684495 **and** 4-hour flow pressure > 0.058576462
- `f5__flow_price__price_response_per_flow__1d_leaf_2` ? 1-day price displacement <= -0.018646787 **and** 1-day price displacement <= -0.03172141
- `f5__flow_price__price_response_per_flow__1d_leaf_3` ? 1-day price displacement <= -0.018646787 **and** 1-day price displacement > -0.03172141
- `f5__flow_price__price_response_per_flow__1d_leaf_5` ? 1-day price displacement > -0.018646787 **and** 1-day price displacement <= 0.013005814
- `f5__flow_price__price_response_per_flow__1d_leaf_6` ? 1-day price displacement > -0.018646787 **and** 1-day price displacement > 0.013005814
- `f5__liq_oi__oi_response_per_liq__4h_leaf_2` ? 4-hour leverage unwind <= -7.6988855e-05 **and** 4-hour liquidation activity severity <= 901535.95
- `f5__liq_oi__oi_response_per_liq__4h_leaf_3` ? 4-hour leverage unwind <= -7.6988855e-05 **and** 4-hour liquidation activity severity > 901535.95
- `f5__liq_oi__oi_response_per_liq__4h_leaf_5` ? 4-hour leverage unwind > -7.6988855e-05 **and** 4-hour liquidation activity severity <= 1438824.1
- `f5__liq_oi__oi_response_per_liq__4h_leaf_6` ? 4-hour leverage unwind > -7.6988855e-05 **and** 4-hour liquidation activity severity > 1438824.1
- `f5__liq_price__liq_price_alignment__4h_leaf_2` ? 4-hour liquidation side pressure <= 0.74430451 **and** 4-hour liquidation side pressure <= -0.68368392
- `f5__liq_price__liq_price_alignment__4h_leaf_3` ? 4-hour liquidation side pressure <= 0.74430451 **and** 4-hour liquidation side pressure > -0.68368392
- `f5__liq_price__liq_price_alignment__4h_leaf_5` ? 4-hour liquidation side pressure > 0.74430451 **and** 4-hour liquidation side pressure <= 0.87310836
- `f5__liq_price__liq_price_alignment__4h_leaf_6` ? 4-hour liquidation side pressure > 0.74430451 **and** 4-hour liquidation side pressure > 0.87310836
- `f5__price_vol__price_displacement_to_vol__4h_leaf_2` ? 4-hour price move retention <= 0.51520514 **and** 4-hour price move retention <= 0.27362023
- `f5__price_vol__price_displacement_to_vol__4h_leaf_3` ? 4-hour price move retention <= 0.51520514 **and** 4-hour price move retention > 0.27362023
- `f5__price_vol__price_displacement_to_vol__4h_leaf_5` ? 4-hour price move retention > 0.51520514 **and** 4-hour price move retention <= 0.71350229
- `f5__price_vol__price_displacement_to_vol__4h_leaf_6` ? 4-hour price move retention > 0.51520514 **and** 4-hour price move retention > 0.71350229
- `f5z__flow_price__flow_price_alignment_percentile__4h_leaf_2` ? 4-hour flow pressure <= -0.043094165 **and** 4-hour flow pressure <= -0.058685091
- `f5z__flow_price__flow_price_alignment_percentile__4h_leaf_3` ? 4-hour flow pressure <= -0.043094165 **and** 4-hour flow pressure > -0.058685091
- `f5z__flow_price__flow_price_alignment_percentile__4h_leaf_5` ? 4-hour flow pressure > -0.043094165 **and** 4-hour flow pressure <= 0.03619677
- `f5z__flow_price__flow_price_alignment_percentile__4h_leaf_6` ? 4-hour flow pressure > -0.043094165 **and** 4-hour flow pressure > 0.03619677
- `f5z__liq_oi__oi_response_per_liq_percentile__4h_leaf_2` ? 4-hour leverage unwind <= -5.7327038e-05 **and** 4-hour liquidation activity severity <= 397266.74
- `f5z__liq_oi__oi_response_per_liq_percentile__4h_leaf_3` ? 4-hour leverage unwind <= -5.7327038e-05 **and** 4-hour liquidation activity severity > 397266.74
- `f5z__liq_oi__oi_response_per_liq_percentile__4h_leaf_5` ? 4-hour leverage unwind > -5.7327038e-05 **and** 4-hour liquidation activity severity <= 946828.94
- `f5z__liq_oi__oi_response_per_liq_percentile__4h_leaf_6` ? 4-hour leverage unwind > -5.7327038e-05 **and** 4-hour liquidation activity severity > 946828.94
- `f5z__liq_price__liq_price_alignment_percentile__4h_leaf_2` ? 4-hour liquidation side pressure <= -0.80964574 **and** 4-hour price displacement <= -0.0019370052
- `f5z__liq_price__liq_price_alignment_percentile__4h_leaf_3` ? 4-hour liquidation side pressure <= -0.80964574 **and** 4-hour price displacement > -0.0019370052
- `f5z__liq_price__liq_price_alignment_percentile__4h_leaf_5` ? 4-hour liquidation side pressure > -0.80964574 **and** 4-hour liquidation side pressure <= 0.74385064
- `f5z__liq_price__liq_price_alignment_percentile__4h_leaf_6` ? 4-hour liquidation side pressure > -0.80964574 **and** 4-hour liquidation side pressure > 0.74385064
- `f5z__price_vol__price_displacement_to_vol_percentile__4h_leaf_2` ? 4-hour price move retention <= 0.4893655 **and** 4-hour price move retention <= 0.27362024
- `f5z__price_vol__price_displacement_to_vol_percentile__4h_leaf_3` ? 4-hour price move retention <= 0.4893655 **and** 4-hour price move retention > 0.27362024
- `f5z__price_vol__price_displacement_to_vol_percentile__4h_leaf_5` ? 4-hour price move retention > 0.4893655 **and** 4-hour price move retention <= 0.69920392
- `f5z__price_vol__price_displacement_to_vol_percentile__4h_leaf_6` ? 4-hour price move retention > 0.4893655 **and** 4-hour price move retention > 0.69920392
- `flow_price_agreement` ? 1d signed flow pressure times sign of 1d displacement.
- `forced_flow_unwind` ? 4h signed liquidation pressure times 4h leverage unwind.
- `move_per_range` ? 1d displacement / positive 1d excursion range.
- `move_per_volatility` ? 1d displacement / sqrt(positive 1d realized variance).
- `positive_days_7d` ? Share positive across seven complete prior daily displacement readings at t-24h through t-168h.
- `pressure_change_72h` ? Current 1d pressure minus its reading exactly 72h earlier.
- `share_acceleration` ? Four-hour variance share less its value exactly four hours ago.
- `volatility_share` ? Incumbent 4h realized variance / 1d realized variance.
- `taker_imbalance_15m` ? (Mean taker buy quote volume minus mean taker sell quote volume) divided by their sum in the completed 15-minute bucket. Each minute series needs at least 12 valid observations and the sum must be positive.
- `imbalance_shift_two_15m_buckets` ? Current 15-minute imbalance minus the immediately preceding 15-minute imbalance.
- `log_vwap_shift_two_15m_buckets` ? Natural log of the current mean of minute cross-venue trade VWAPs minus the log of the previous bucket mean. Each mean uses at least 12 minute values.
- `flow_price_alignment_15m` ? Current 15-minute taker imbalance multiplied by the sign of the change in log mean trade VWAP between the two buckets.

## replay_frozen_15m_direction.py / rule_leaves

```python
def rule_leaves(frame):
    rules=pd.read_csv(SOURCE.parent/'pattern_rules.csv')
    leaves={}
    for teacher,tree in rules[rules.retained].groupby('teacher'):
        nodes=tree.set_index('node')
        pending=[(0,np.ones(len(frame),dtype=bool))]
        while pending:
            node,mask=pending.pop()
            if node not in nodes.index:
                leaves[f'{teacher}_leaf_{node}']=mask.astype('int8')
                continue
            row=nodes.loc[node]
            observed=frame[row.feature].to_numpy()
            pending.append((int(row.left), mask & (observed<=row.threshold)))
            pending.append((int(row.right),mask & (observed>row.threshold)))
    return pd.DataFrame(leaves,index=frame.index)
```

## replay_frozen_15m_direction.py / build_design_from_history

```python
def build_design_from_history(history,short):
    """Combine guarded F4/F4Z/F2 and four native short inputs indexed by UTC timestamp."""
    history=history.sort_index().asfreq('30min')
    if not isinstance(history.index,pd.DatetimeIndex) or not isinstance(short.index,pd.DatetimeIndex):
        raise TypeError('Inputs must have a UTC datetime index.')
    x=history.filter(regex='^f4z?__').copy()
    x=x.join(rule_leaves(x))
    p=history['f4__flow__flow_pressure__1d']
    move=history['f4__price__price_displacement__1d']
    v=history['f4__volatility__realized_variance_level__1d']
    def ratio(a,b):
        return a/b.where(b>0)
    def m(num,w='4h'):
        return history[f'f2__m{num:03d}__{w}']
    x['move_per_volatility']=ratio(move,np.sqrt(v.where(v>0)))
    x['move_per_range']=ratio(move,history['f4__price__price_excursion_range__1d'])
    x['flow_price_agreement']=p*np.sign(move)
    x['forced_flow_unwind']=history['f4__liquidation__liq_side_pressure__4h']*history['f4__oi__leverage_unwind__4h']
    x['carry_agreement']=np.sign(history['f4__funding__funding_level__4h'])*np.sign(history['f4__basis__basis_level__4h'])
    x['pressure_change_72h']=p-p.shift(144)
    prior=pd.concat([move.shift(48*d) for d in range(1,8)],axis=1)
    x['positive_days_7d']=prior.gt(0).mean(axis=1).where(prior.notna().all(axis=1))
    vol_share=ratio(history['f4__volatility__realized_variance_level__4h'],v)
    x['volatility_share']=vol_share
    x['share_acceleration']=vol_share-vol_share.shift(8)
    turn=m(11)+m(12)
    turn_1d=m(11,'1d')+m(12,'1d')
    oi_change=ratio(m(18)-m(17),m(17))
    x['f2_taker_share']=ratio(turn,turn_1d)
    x['f2_taker_balance']=ratio(m(11)-m(12),turn)
    x['f2_liq_share']=ratio(m(3),m(3,'1d'))
    x['f2_liq_peak_fraction']=ratio(m(8),m(3))
    x['f2_liq_concentration']=m(7)
    x['f2_oi_change']=oi_change
    x['f2_excursion_balance']=ratio(m(26)+m(27),m(26)-m(27))
    x['f2_range_spike']=ratio(m(34),m(33))
    x['f2_funding_change']=m(39)
    x['f2_basis_change']=m(46)
    x['f2_variance_turnover_gap']=vol_share-ratio(turn,turn_1d)
    x['f2_oi_price_alignment']=oi_change*np.sign(m(25))
    return x.join(short).replace([np.inf,-np.inf],np.nan)
```

## replay_frozen_15m_direction.py / apply_direction

```python
def apply_direction(archive,issue,model,x):
    filename=f'Rebuilt/short_direction/15m_{issue.date()}_{model.replace("/","-")}.joblib'
    state=joblib.load(BytesIO(archive.read(filename)))
    assert state['issue']==issue and state['model']==model
    frame=x[state['columns']].clip(state['low'],state['high'],axis=1)
    with threadpool_limits(limits=2):
        prob=state['estimator'].predict_proba(frame)[:,1]
    prob=np.where(frame.notna().any(axis=1), prob, state['prior'])
    return np.clip(prob*state['probability_scale']+state['probability_offset'],1e-6,1-1e-6)
```

## frozen_15m_micro_inputs.py / frozen_micro_inputs

```python
def frozen_micro_inputs(start='2024-03-31', end='2025-01-23'):
    archive_path = ROOT/'Datasets/HANDOFF_4_ML_RESEARCH_V1_MINUTE.zip'
    prefix = 'HANDOFF_4_ML_RESEARCH_V1/'
    frames = []
    with ZipFile(archive_path) as archive, archive_path.open('rb') as raw, duckdb.connect() as db:
        db.execute("SET TimeZone='UTC'")
        db.execute("SET threads=2")
        manifest = json.loads(archive.read(prefix+'catalog/X_FILES.json'))
        hashes = {r['path']:r['sha256'] for r in manifest if r.get('asset')=='BTCUSDT' and r.get('layer')=='F1'
                  and r.get('family') in ('FLOW','PRICE')}
        for path in sorted({m.split('!')[0] for m in hashes}):
            if not start <= Path(path).stem <= end:
                continue
            parts = []
            with ZipFile(StoredZipView(raw,archive.getinfo(prefix+path))) as shard:
                for family,codes in [('FLOW',['f060','f061']),('PRICE',['f066'])]:
                    key = f'FORGE/F1/1m/BTCUSDT/{family}.parquet'
                    payload = shard.read(key)
                    if hashlib.sha256(payload).hexdigest()!=hashes[path+'!'+key]:
                        raise ValueError(f'Hash mismatch for {path}!{key}')
                    fields = [f'CASE WHEN source_structurally_valid AND {c}__is_valid AND isfinite({c}__value) '
                              f'THEN {c}__value::DOUBLE END AS {c}' for c in codes]
                    minute = db.read_parquet(BytesIO(payload)).filter("canonical_symbol='BTCUSDT' AND available_after<=window_end").select(
                        'window_end,'+','.join(fields)).df().set_index('window_end',verify_integrity=True).sort_index()
                    rule = dict(closed='right',label='right')
                    bins = minute.resample('15min',**rule)
                    quarter = bins.mean().where(bins.count()>=12)
                    t = minute.resample('30min',**rule).size().index
                    if family=='FLOW':
                        total = quarter.f060+quarter.f061
                        imbalance = (quarter.f060-quarter.f061)/total.where(total>0)
                        packed = pd.DataFrame(dict(taker_imbalance_15m=imbalance,
                            imbalance_shift_two_15m_buckets=imbalance.diff()))
                    else:
                        logmove = np.log(quarter.f066.where(quarter.f066>0)).diff()
                        packed = logmove.to_frame('log_vwap_shift_two_15m_buckets')
                    parts.append(packed.reindex(t))
            frames.append(pd.concat(parts,axis=1))
    result = pd.concat(frames).sort_index()
    if result.index.has_duplicates:
        repeats = result.loc[result.index.duplicated(keep=False)]
        assert not repeats.groupby(level=0).nunique().gt(1).any().any()
        result = result.groupby(level=0).first()
    result['flow_price_alignment_15m'] = result.taker_imbalance_15m*np.sign(result.log_vwap_shift_two_15m_buckets)
    result.index.name = 'decision_time'
    return result.replace([np.inf,-np.inf],np.nan)
```
