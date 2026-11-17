"""Export existing saved predictions only; no fitting, inference or evaluation.

Python standard library only. The default export is pinned to two completed runs.
Run from Week 2: python Code/export_conditional_mean_handoff.py
"""
from pathlib import Path
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import argparse
import csv
import hashlib
import json
import shutil
import zipfile

FIELDS = ['timestamp_utc', 'symbol', 'horizon_hours', 'mean_return_if_up_bps', 'mean_return_if_down_bps']
UP_RUN = 'mlp_rf_conditional_means_48h_v1/20261112T064753697944Z'
DOWN_RUN = 'conditional_means_signed_48h_v1/20261112T063535834150Z'
UP_SHA = 'd21085717cacdabdfcc77aeda839ae55425e4d55382b18aaa89916965addcdb5'
DOWN_SHA = 'ed889ac783630a985a6b21c82dfa4f49d659577a7236af98eace2cbf1e8aa48b'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Timestamp must have timezone')
    return dt.astimezone(timezone.utc)


def iso(value):
    return value.isoformat(timespec='seconds').replace('+00:00', 'Z')


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def read_run(folder, expected_hash):
    complete = json.loads((folder / 'COMPLETE.json').read_text())
    path = folder / 'signed_return_predictions.csv'
    if complete.get('status') != 'COMPLETE' or sha(path) != expected_hash or complete['prediction_sha256'] != expected_hash:
        raise ValueError(f'Run is incomplete or predictions changed: {folder}')
    if sha(folder / 'CONTRACT.json') != complete['contract_sha256']:
        raise ValueError(f'Contract hash mismatch: {folder}')
    rows = {}
    with path.open(newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            t = stamp(row['decision_time'])
            if t in rows or row['symbol'] != 'BTCUSDT' or int(row['horizon_hours']) != 48:
                raise ValueError('Duplicate timestamp or wrong symbol/horizon')
            if stamp(row['forecast_end']) != t + timedelta(hours=48) or stamp(row['issue']) > t:
                raise ValueError('Wrong endpoint or future model issuance')
            rows[t] = row
    return rows


def mean_value(row, field, negative=False):
    if row is None:
        return '', 'timestamp absent from saved prediction population'
    value = row.get(field, '').strip()
    if not value:
        return '', 'saved prediction blank'
    number = Decimal(value)
    if not number.is_finite():
        return '', 'saved prediction nonfinite'
    if number < 0:
        raise ValueError(f'Expected nonnegative stored magnitude: {field}')
    return (format(-number, 'f') if negative else value), ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    out = args.output_dir or workspace / 'Deliverables/BTCUSDT_48h_conditional_means_2026-11-12'
    archive = out.with_suffix('.zip')
    if out.exists() or archive.exists():
        raise FileExistsError('Preserving existing export; use --output-dir with a new path')
    up_folder, down_folder = workspace / 'Code/experiments' / UP_RUN, workspace / 'Code/experiments' / DOWN_RUN
    up, down = read_run(up_folder, UP_SHA), read_run(down_folder, DOWN_SHA)
    if set(up) != set(down):
        raise ValueError('Source timestamp populations differ; investigate before export')
    for t in up:
        if stamp(up[t]['issue']) != stamp(down[t]['issue']):
            raise ValueError('Source fold issuance dates differ')
    selection = json.loads((up_folder / 'SELECTIONS.json').read_text())
    up_audit = json.loads((up_folder / 'FIT_AUDIT.json').read_text())
    down_audit = json.loads((down_folder / 'FIT_AUDIT.json').read_text())
    ledger = []
    for selected in selection:
        if selected['side'] != 'UP' or selected['policy'] != 'rf':
            continue
        issue = stamp(selected['issue'])
        ua = next(a for a in up_audit if stamp(a['issue']) == issue and a['side'] == 'UP'
                  and a['family'] == 'rf' and a['pack'] == selected['pack'])
        da = next(a for a in down_audit if stamp(a['issue']) == issue)
        if not ua['replay_passed'] or not da['replay_passed']:
            raise ValueError('Source model replay audit failed')
        if stamp(ua['max_training_label_end']) >= issue or stamp(da['down_max_label_end']) >= issue:
            raise ValueError('Training labels overlap model issuance')
        if sha(up_folder / ua['artifact']) != ua['sha256']:
            raise ValueError('UP model hash mismatch')
        down_model = f'{issue:%Y-%m}_mean_models.joblib'
        if sha(down_folder / down_model) != da['model_sha256']:
            raise ValueError('DOWN model hash mismatch')
        ledger.append(dict(issue_utc=iso(issue), up_family='squared_error_random_forest',
            up_feature_pack=selected['pack'], up_training_rows=ua['training_rows'],
            up_max_training_label_end_utc=iso(stamp(ua['max_training_label_end'])),
            up_model_file=ua['artifact'], up_model_sha256=ua['sha256'],
            down_family='ridge_plus_squared_error_random_forest', down_rf_weight=da['down_rf_weight'],
            down_training_rows=da['down_rows'], down_max_training_label_end_utc=iso(stamp(da['down_max_label_end'])),
            down_model_file=down_model, down_model_sha256=da['model_sha256']))
    ledger.sort(key=lambda r: r['issue_utc'])
    if len(ledger) != 4:
        raise ValueError('Expected four quarterly model entries')
    start, end = stamp('2024-01-01T00:00:00Z'), stamp('2025-01-22T23:30:00Z')
    output, missing = [], []
    t = start
    while t <= end:
        uv, ur = mean_value(up.get(t), 'UP_rf_mean_bps')
        dv, dr = mean_value(down.get(t), 'down_mean_magnitude_bps', negative=True)
        output.append(dict(zip(FIELDS, [iso(t), 'BTCUSDT', '48', uv, dv])))
        if ur or dr:
            missing.append(dict(timestamp_utc=iso(t), up_missing_reason=ur, down_missing_reason=dr))
        t += timedelta(minutes=30)
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'conditional_mean_predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output)
    with (out / 'missing_predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['timestamp_utc', 'up_missing_reason', 'down_missing_reason'])
        writer.writeheader()
        writer.writerows(missing)
    with (out / 'training_refit_ledger.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)
    # Verify the delivered CSV, not only the in-memory construction.
    with (out / 'conditional_mean_predictions.csv').open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError('Export schema mismatch')
        written = list(reader)
    if written != output:
        raise ValueError('Export readback changed values or order')
    if len({r['timestamp_utc'] for r in written}) != len(written):
        raise ValueError('Duplicate exported timestamp')
    for r in written:
        t = stamp(r['timestamp_utc'])
        if r['mean_return_if_up_bps']:
            assert Decimal(r['mean_return_if_up_bps']) == Decimal(up[t]['UP_rf_mean_bps'])
        if r['mean_return_if_down_bps']:
            assert Decimal(r['mean_return_if_down_bps']) == -Decimal(down[t]['down_mean_magnitude_bps'])
    available = [r for r in written if r['mean_return_if_up_bps'] and r['mean_return_if_down_bps']]
    provenance = out / 'source_configuration'
    provenance.mkdir()
    for prefix, folder in [('up_rf', up_folder), ('down_hybrid', down_folder)]:
        for name in ['CONTRACT.json', 'COMPLETE.json', 'FIT_AUDIT.json']:
            shutil.copyfile(folder / name, provenance / f'{prefix}_{name}')
    shutil.copyfile(up_folder / 'SELECTIONS.json', provenance / 'up_rf_SELECTIONS.json')
    scripts = out / 'training_scripts'
    scripts.mkdir()
    # Copy the exact code snapshots retained with the completed runs.
    shutil.copyfile(up_folder / 'source_fit_mlp_rf_conditional_means_48h.py', scripts / 'fit_mlp_rf_conditional_means_48h.py')
    shutil.copyfile(up_folder / 'source_fit_conditional_48h_mean_features.py', scripts / 'fit_conditional_48h_mean_features.py')
    shutil.copyfile(down_folder / 'source_snapshot.py', scripts / 'fit_conditional_means_signed_48h.py')
    shutil.copyfile(Path(__file__), out / 'export_conditional_mean_handoff.py')
    package = workspace.parent / 'Week 1/Dataset/TEAM_48H_PLAYGROUND_V1/TEAM_48H_PLAYGROUND_V1'
    shutil.copyfile(package / 'CONTRACT.json', provenance / 'original_target_CONTRACT.json')
    shutil.copyfile(workspace.parent / 'Week 1/Code/canonical_return_features.py', provenance / 'canonical_return_features.py')
    info = dict(exported_at_utc=iso(datetime.now(timezone.utc)), policy_id='UP_RF_MEAN__DOWN_HYBRID_MEAN_V1',
        freeze_status='Fixed export of retrospective walk-forward predictions; family pairing selected after historical evaluation, not prospective/live freeze',
        target='conditional mean ordinary 48-hour endpoint return', model_estimates_not_calibration_guarantee=True,
        reference_price='historical canonical f066 reference price; exchange/price type unresolved from inspected source contract',
        schema=FIELDS, up_source_column='UP_rf_mean_bps', down_source_column='down_mean_magnitude_bps',
        transform='UP copied exactly; DOWN positive ordinary-bps magnitude negated; no unit conversion of predictions',
        missing_policy='Blank, never replaced by zero; no baseline fallback',
        requested_first_utc=iso(start), requested_last_utc=iso(end), cadence_minutes=30,
        exported_grid_rows=len(written), complete_prediction_rows=len(available), missing_rows=len(missing),
        first_complete_timestamp_utc=available[0]['timestamp_utc'], last_complete_timestamp_utc=available[-1]['timestamp_utc'],
        source_population_rows=len(up), source_first_utc=iso(min(up)), source_last_utc=iso(max(up)),
        source_runs=dict(up=UP_RUN, down=DOWN_RUN), source_predictions_sha256=dict(up=UP_SHA, down=DOWN_SHA),
        checks=dict(source_completion_and_hashes=True, exact_source_timestamp_alignment=True,
                    model_hashes=True, source_replay_audits_checked=True, cutoff_purge=True,
                    five_column_schema=True, unique_regular_grid=True, csv_roundtrip=True,
                    every_available_value_matches_source=True, future_outcomes_used_for_export=False),
        execution='Export and validation only; no fitting, prediction rerun, baseline filling or market-data fetch')
    dump(out / 'export_configuration.json', info)
    ledger_text = '\n'.join(f"{r['issue_utc']}: UP RF {r['up_feature_pack']} features; DOWN RF weight {r['down_rf_weight']}" for r in ledger)
    readme = f'''BTCUSDT 48-hour conditional mean endpoint-return handoff

1. Files and coverage
conditional_mean_predictions.csv has exactly the five requested columns.
UTC half-hour grid: {iso(start)} through {iso(end)}, inclusive.
Grid rows: {len(written)}. Rows with both predictions: {len(available)}.
Available predictions: {available[0]['timestamp_utc']} through {available[-1]['timestamp_utc']}.
Rows with missing predictions: {len(missing)}. See missing_predictions.csv.
Absent source timestamps are retained as blank predictions, not zero. No inference
or interpolation was performed for gaps. Source eligibility reasons for internal
gaps have not been independently reconstructed. Rows after the source's final
timestamp are beyond available coverage. Both side predictions are supplied at
every available source timestamp; future realized direction is never used to
choose, filter, or replace either side during export.

2. Outcome and units
For forecast origin t, r = 10000 * (P(t+48h)/P(t) - 1), ordinary basis points.
UP means r > 0. DOWN means r < 0. Flat outcomes are excluded from side training;
the source evaluation runners reject flats in their development population.
mean_return_if_up_bps estimates E[r | information at t, r > 0].
mean_return_if_down_bps estimates E[r | information at t, r < 0] and is NEGATIVE.
The source models train on absolute ordinary-bps endpoint returns separately by
realized historical side. Targets were converted observation by observation from
stored signed log bps with 10000*expm1(log_bps/10000) BEFORE fitting.
The DOWN source is positive magnitude; export negates it. UP values are copied
unchanged. Predictions are not log bps, medians, maximum excursions, or error
metrics. The squared-error objective targets conditional averages; calibration
and predictive accuracy are not guaranteed by the objective alone.
100 bps equals one percentage point. No baseline averages substitute for models.

3. Reference price
The original package CONTRACT.json identifies P as the 'historical canonical
f066 reference price'. The exact exchange, construction and price type (close,
mid, mark, last or aggregation) are NOT established by the inspected contract.
Do not assume an executable trade price. The prepared-data contract records a
prior audit matching 48-hour log targets to sums of 576 canonical five-minute
log returns; this export does not rerun that audit or reconstruct raw prices.
Source configuration includes the target contract and canonical-return helper.
To resolve the remaining definition, request the canonical f066 label-builder
script/configuration and reference-price specification from its data owner,
including price timestamps, exchange/aggregation and missing-price treatment.

4. Model policy and refits
UP: saved squared-error Random Forest, 128 trees, max depth 3, min leaf128,
bootstrap4096, seed48. Its 4/5/8 feature pack was selected separately each fold
using preceding validation MSE. Uses UP_rf_mean_bps, not the selected MLP/RF column.
DOWN: saved Ridge(alpha1) plus squared-error RF with the same RF capacity and
eight features. RF blend weight selected by row-weighted MSE across earlier
validation cutoffs 9/6/3 months before issuance. Weight zero means fitted Ridge
alone; it is still a feature-based learned model, not a constant side average.
{ledger_text}
These fits use expanding historical data with label_end STRICTLY before issue.
Training-row counts, latest training outcome times, artifact IDs and SHA256
identities are in training_refit_ledger.csv. An origin's outcome must have fully
matured before it can enter a later fit. Model weights/preprocessing are fixed
within each quarterly evaluation block. October's fit serves the final block
through the available January 2025 endpoint; there was no January 2025 refit.
The UP and DOWN policies use historical actual sides only to form training sets.
At inference, each model predicts every origin using its input features.

5. Freeze and forecast availability
This is a fixed, hash-identified export of retrospectively generated walk-forward
predictions from completed November 2026 runs. It is NOT evidence that predictions
were generated live in 2024/2025. The UP-RF/DOWN-hybrid family pairing was proposed
after inspecting historical development scores; it is not a prospectively frozen,
untouched-holdout policy. This selection exposure must follow downstream reports.
timestamp_utc is the forecast origin; issue is the simulated fit/refit cutoff.
No actual historical publication/arrival timestamps are recorded for these
predictions. At-origin availability is assumed by the research setup, not proven
by a live availability audit. The original package retains PROVISIONAL source
timing limitations and unresolved funding-boundary construction. Derived price
features use canonical five-minute return endpoints strictly before origin;
that does not independently certify all other feature availability.

6. Export and configuration scripts
export_conditional_mean_handoff.py reproduces this export using Python's standard
library and the existing workspace source CSVs/artifacts. It performs no training.
Example: python export_conditional_mean_handoff.py --workspace "PATH_TO_WEEK_2" --output-dir "NEW_EXPORT_FOLDER"
Use a new output path; existing exports are never overwritten. The script is
pinned to the source prediction hashes and run IDs in export_configuration.json.
training_scripts contains the exact existing training-code snapshots; source
configuration contains contracts, completion and replay-audit records and UP
feature-pack selections. Training scripts still need the original Week 1 data,
helper modules and installed dependencies; the package is not a standalone model
retraining environment and does not include weights or raw data.

7. Verification
Source completion records, contracts, model hashes and saved replay audits were
checked. Export was reopened and all values compared to source values, allowing
only the declared DOWN sign change. Schema, unique grid, horizons, issuance and
training-label cutoffs were checked. Missing rows are explicitly listed.
CHECKSUMS.json records each packaged file's SHA256. No new performance, calibration
or fresh-data claim is made by this export.
'''
    (out / 'README.txt').write_text(readme, encoding='utf-8')
    dump(out / 'CHECKSUMS.json', {str(p.relative_to(out)).replace('\\', '/'): sha(p)
                                for p in sorted(out.rglob('*')) if p.is_file()})
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                z.write(path, str(path.relative_to(out)))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP integrity check failed')
    print(json.dumps(info, indent=2))
    print(f'Export: {out}\nArchive: {archive}')


if __name__ == '__main__':
    main()
