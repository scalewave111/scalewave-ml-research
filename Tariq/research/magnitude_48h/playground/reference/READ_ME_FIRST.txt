FORGE COMBINED RESEARCH PACKAGE — EARLIEST 90%

Coverage: 1,008 daily partitions, 2022-04-22 through 2025-01-23 inclusive.
Source snapshot: 1,120 contiguous days, 2022-04-22 through 2025-05-15.
Held back from BOTH features and prices: 112 days, 2025-01-24 through 2025-05-15.
This adds 307 days to the two previously supplied packages.

CONTENTS
FORGE_PRO_V1/dates/: original directory layout; all seven F4/F4Z families,
BTCUSDT/ETHUSDT/SOLUSDT, one-minute source clock. 42 files/day; 42,336 files.
prices/: one combined minute-price Parquet per symbol, 1,451,520 rows each.
reference/: unchanged formulas, dictionary, historical cuts, state rules and framework.
tools/build_y_1h.py: unchanged helper to build separate one-hour targets.
PACKAGE_MANIFEST.json: exact date split, scope, price summaries and source identities.
SOURCE_DAY_RECEIPTS.json: selected-day certification and upstream identity summaries.
SOURCE_PRICE_MANIFEST.json: source checksums for the 3,024 input price files.
PRICE_VERIFICATION.json: price verification scope.
FILES.sha256: checksums for all other files in this archive.
tools/verify_package.py: verify the extracted package, requiring Python only.

All data is directly inside this ONE ZIP; no nested archives.

PRICE AND JOIN SEMANTICS
f066__value is minute cross-venue trade VWAP, decimal128(38,18), USDT-equivalent
per coin; it is not last trade, candle close, or an execution fill.
decision_time = window_end, UTC. Join features and prices on
(canonical_symbol, decision_time), with one-to-one key validation.
source_date preserves the input daily partition. The last shared minute ends at
2025-01-24 00:00 UTC but belongs to the 2025-01-23 partition; no held-back partition
is supplied. Preserve validity and available_after <= decision_time separately
for every feature. Never fill missing/invalid measurements with zero.

Price admission: f066__is_valid, positive finite price, source_structurally_valid,
and available_after <= decision_time. Capture certification is a separate field.
No prices are rounded, filled, interpolated or resampled.

TARGET HELPER
python tools/build_y_1h.py --prices prices/PRICE_1m_BTCUSDT.parquet --output labels/Y_1h_BTCUSDT.parquet
Repeat for ETHUSDT and SOLUSDT. Requires numpy, pandas, pyarrow.
Y = ln(P(t+60m))-ln(P(t)); log basis points multiply by 10,000.
The helper requires all 61 minute prices from t through t+60 to pass admission.
The final 60 rows cannot have complete labels within this release. No extra
held-back prices are supplied for label completion. Keep target columns out of X;
exclude training labels that reach into evaluation periods.

REFERENCE AND PROVENANCE
Historical PRICE/FLOW cuts were fitted on 2022-10-07 through 2024-02-01. They are
for reproduction and are not automatically train-only cuts for a new experiment.
Reference documents retain their historical scope; PACKAGE_MANIFEST.json defines
this expanded release. The 10% exclusion does not certify an untouched holdout.
Source binding identities change within the supplied history; the manifest records
these identities explicitly. Original source values are preserved.

VERIFY AFTER EXTRACTION
python tools/verify_package.py
The external .zip.sha256 authenticates the complete ZIP.
