from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def read(n):return json.loads((ROOT/n).read_text(encoding='utf-8-sig'))
ledger=read('LEDGER_AUDIT.json');replay=read('FULL_REPLAY_AND_CAUSALITY.json');inputs=read('INPUT_RECONSTRUCTION.json')
retrain=read('Q4_RETRAINING_LGB470.json');native=read('NATIVE_FEATURE_PROBES.json');runtime=read('ADDITIONAL_RUNTIME_CHECKS.json')
all_entries=ledger['all_entries_primary'];diagnostic=ledger['single_position_diagnostic_only']
max_error=max(v for q in replay['replay'] for v in q['max_absolute_errors'].values())
assert replay['status']=='PASS' and ledger['native_target_check']['valid']
summary={'decision':'HISTORICAL_EXECUTION_AND_METRICS_VERIFIED; LIVE_EDGE_NOT_ESTABLISHED','primary_strategy':'All eligible entries; multiple concurrent positions permitted by user',
         'package_files_verified':76,'model_files_loaded':14,'frozen_replay_rows':replay['replay_rows'],'max_replay_numeric_error':max_error,
         'primary_metrics':all_entries,'tests':{'bundled':3,'additional_runtime':runtime['tests_passed'],'native_feature_probe_cells':native['checked_cells'],'future_label_perturbation':'PASS','prefix_causality':'PASS'},
         'q4_retraining_status':retrain['status'],'monthly_file':'MONTHLY_TRADES.csv','no_h4a_fitting_or_evaluation':True,
         'limitations':['Iterative selection on the same historical quarters','Gross reference-VWAP returns; actual fills/costs/capacity not measured','Frozen routing ends before 2025-01-01','Capital/position sizes unspecified; summed entry bps is not account ROI','All seven training reruns were not independently repeated in the original environment']}
(ROOT/'AUDIT_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
lines=['# Sakthi SAK_GROSS_V01 independent verification — 26 November 2026','',
       '**The supplied historical model runs, and its headline result is reproducible.** The primary strategy takes every qualifying entry, with multiple concurrent trades allowed as the user specified. All 1,145 eligible entries, including 390 losses, give 755 wins (65.94%) and **24.761058 gross bps per entry**. This establishes historical reproducibility; a fresh validation and measured execution are still needed for a live profitability claim.','',
       '## Checks performed','',
       '| Check | Result |','|---|---|',
       '| ZIP and supplied file hashes | CRC passed; all 76 manifest-listed files matched; all 14 native weight files loaded |',
       '| Original interfaces | Direction, magnitude and combined files match the earlier pinned GitHub snapshot |',
       '| Example and input-contract checks | 33 example rows matched exactly; 3 bundled tests and '+str(runtime['tests_passed'])+' additional runtime checks passed |',
       '| Full independent input reconstruction | 1,418,400 × 98 matrix rebuilt from existing H4 arena; original calendar matched the APEX index |',
       '| Full frozen inference | All 923,007 origins replayed; maximum numeric discrepancy '+str(max_error)+'; selections/sides/timestamps matched |',
       '| Native targets | All origins matched native H4 Y(t+1); independent native-price formula discrepancy below 2.3e-12 bps |',
       '| Native feature probes | All 86 source fields on six fixed days: 743,040 cells matched exactly; availability/validity guards passed |',
       '| Timing | Actual seven-fold training/calibration purges and frozen priors matched; immature/future-label perturbation and derived-prefix check passed |',
       '| Bounded retraining | Latest 2024Q4 pair retrained with fixed settings: '+retrain['status']+'; see receipt |','',
       'The source ZIP and its extracted code/weights were kept unchanged. Inputs and replay outputs were created in this audit folder. No Handoff 4a data was fitted or evaluated and no remote model was changed. The reconstructed X.npy matches the supplied SHA256 exactly. The newly written aligned_panel.parquet has a different byte hash, so full file-byte parity is not claimed for that label file; every evaluation target matches exactly and all seven split counts/priors were checked against native labels.','',
       '## Monthly trade classification — all entries','',
       'Win/loss uses the sign of the realised gross return. This table includes losses and months with no entries; it does not filter on the realised 10-bps outcome. All returns below are gross, before fees, spread, slippage and funding.','',
       '| Month | Trades | Long / short | Win / loss | Win rate | Avg gross bps | Sum entry bps |',
       '|---|---:|---:|---:|---:|---:|---:|']
for row in ledger['all_entry_months']:
    win='—' if row['win_rate'] is None else f"{row['win_rate']:.2%}"
    mean='—' if row['avg_gross_bps'] is None else f"{row['avg_gross_bps']:+.2f}"
    lines.append(f"| {row['month']} | {row['trades']} | {row['longs']} / {row['shorts']} | {row['wins']} / {row['losses']} | {win} | {mean} | {row['sum_gross_bps']:+.2f} |")
lines += ['', 'Full classifications, threshold-hit counts, cost scenarios and all-month forecast counts are in [MONTHLY_TRADES.csv](MONTHLY_TRADES.csv). There are no breakeven selected entries.','',
          'The strategy is selective and strongly long-biased: 1,081 longs and 64 shorts (94.4% long), spread over only 78 active days. Five calendar months have zero trades. March and August 2024 contain 680 of 1,145 entries (59.4%). April 2024 averages -1.76 gross bps despite a 62.7% win rate. August has 389 entries averaging +1.09 gross bps; at an assumed 2-bps round-trip cost this becomes -0.91 bps per entry. The top five gross-profit days account for 41.65% of the pooled gross sum.','',
          '## Multiple entries, capital and costs','',
          'Multiple entries are allowed and are the primary result. The maximum is 15 simultaneous unit positions. A per-entry notional of 1% of initial capital would therefore imply at most 15% gross notional exposure under this benchmark schedule; this is an illustration, not an allocation recommendation. Shared 15-minute price windows make adjacent returns dependent. The 28,351.41 summed entry bps are not a percentage return on account equity.','',
          '| Fixed assumed round-trip cost | Average after subtracting cost, bps per entry |','|---|---:|',
          '| 0 bps | 24.7611 |','| 2 bps | 22.7611 |','| 4 bps | 20.7611 |','| 8 bps | 16.7611 |','',
          'These are arithmetic cost scenarios. Actual liquidity, latency, cross-venue VWAP attainability, fees, slippage and funding have not been measured. The fixed-size realised P&L drawdown at trade exits is '+f"{all_entries['max_realised_drawdown_bps_equal_entry_notional']:.2f}"+' entry-notional bps; it excludes intratrade mark-to-market drawdown and needs position sizing before conversion to account risk.','',
          'For diagnosis only, first-signal-when-flat accounting gives 263 non-overlapping trades, 169 wins and 17.8926 gross bps per trade. This is a different trading policy, not a requirement or a correction that invalidates the 1,145-entry multi-position strategy.','',
          'The supplied 14-day moving-block bootstrap was independently reproduced within floating-point precision: gross-mean 95% interval about [12.28, 44.70] bps, accuracy [60.84%, 74.55%]. These intervals account for some serial dependence and do not correct for repeated model/feature/threshold selection on these quarters.','',
          '## Comparison and remaining limitations','',
          'On the common 923,007 historical origins, Sakthi direction log loss is 0.6900855855 and magnitude MAE is 10.1004419046 bps. The earlier Sid v002 reported 0.6901080684 and 10.0133865577 respectively: Sakthi is slightly better on direction loss and worse on magnitude MAE. These forecast metrics and differently selected trade policies do not establish one universally best model. K80\'s 4.909 uses different selection and drift/overlap accounting; do not rank it directly against 24.761 without a common policy.','',
          'The package deliberately covers 2023-04-01 through 2024-12-31. It rejects 2025 onward, including Handoff 4a, and cannot be used unchanged for today\'s live predictions. A separately frozen later-date routing/training version and an as-of feature/history feed are required. Mature-return features consume only resolved historical outcomes in the audited construction; the live feed must maintain that state correctly. The predictor accepts NaN feature values and can still emit forecasts with every prepared feature missing, so a live completeness/staleness gate is an operational requirement.','',
          'The original environment declared Python 3.13.9 and LightGBM 4.7.0. Frozen replay passed on the existing Python 3.12.8 / LightGBM 4.6.0 environment. Q4 retraining with 4.6.0 changed 23 entry decisions (113 entries instead of 94). A controlled repeat using an isolated official LightGBM 4.7.0 wheel recovered the magnitude forecasts exactly, the same 94 entries, identical calibration and score agreement within 1.5e-14. Keep the supplied dependency pins for training. See [4.6 mismatch](Q4_RETRAINING_AUDIT.json) and [4.7 pass](Q4_RETRAINING_LGB470.json). The global installation was not changed. Other runtime differences are in [RUNTIME_CHECK.json](RUNTIME_CHECK.json); all-seven-quarter retraining in the complete original environment was not repeated here. LightGBM documents that deterministic settings do not guarantee identical training across versions/builds/systems. [Primary documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html#deterministic).','',
          'All native feature values were recovered through the existing H4 arena, whose required selected matrix is content-compared with the delivered identity. Native-source value checking covered six full days, not an exhaustive reconstruction of the original 60-GB feature archive. Model choices were made through iterative research on the same historical quarters. A fresh permitted validation remains necessary; the final-test block must stay untouched until the complete recipe is frozen.','',
          '## Evidence and rerun','',
          '- [Independent input reconstruction](INPUT_RECONSTRUCTION.json), [native probes](NATIVE_FEATURE_PROBES.json), [price audit](PRICE_TARGET_AUDIT.json).',
          '- [Full replay and causality](FULL_REPLAY_AND_CAUSALITY.json), [ledger audit](LEDGER_AUDIT.json), [runtime checks](ADDITIONAL_RUNTIME_CHECKS.json).',
          '- [Quarterly table](QUARTERLY_TRADES.csv), [daily table](DAILY_TRADES.csv), [all-entry ledger](ALL_ENTRY_LEDGER.parquet).',
          '- Original package: `source/SAK_GROSS_V01_COMPLETE/`. Independent helpers: `tools/`. Rebuild/replay outputs are retained for traceability.','']
(ROOT/'FINDINGS.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'report':str(ROOT/'FINDINGS.md'),'replay_rows':replay['replay_rows'],'max_error':max_error,'q4_retraining':retrain['status'],'monthly_rows':len(ledger['all_entry_months'])},indent=2))
