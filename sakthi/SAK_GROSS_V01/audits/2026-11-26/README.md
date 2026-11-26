# SAK_GROSS_V01 independent audit — 26 November 2026

The supplied historical package reproduced across **923,007 forecasts**. The primary strategy permits concurrent entries: **1,145 entries, 755 wins, 390 losses, 65.94% win rate and 24.761058 gross bps per entry**. Peak concurrency is 15 positions.

Read the [full findings and monthly table](FINDINGS.md), [monthly CSV](MONTHLY_TRADES.csv), [quarterly CSV](QUARTERLY_TRADES.csv), and [structured summary](AUDIT_SUMMARY.json). The ledger includes every qualifying entry and all losses. Single-position accounting is a diagnostic comparison only.

- Seven frozen quarterly models replayed from independently reconstructed inputs.
- Latest-quarter retraining matched with **LightGBM 4.7.0**; [receipt](Q4_RETRAINING_LGB470.json).
- All 21 UTC months are reported, including five with zero trades.
- April 2024 averages -1.76 gross bps; August's 389 entries average +1.09 gross bps.
- March and August contain 59.4% of entries; 94.4% of all entries are long.

This verifies historical reproducibility. Gross benchmark returns do not establish actual fills, execution costs or account returns. The historical quarters were used in iterative research. The supplied router covers April 2023–December 2024 and rejects 2025 onward. No Handoff 4a fitting or evaluation occurred.

[Source identity and rerun prerequisites](SOURCE_PACKAGE.md) explain the external inputs. [PUBLISH_MANIFEST.json](PUBLISH_MANIFEST.json) records the copied audit files and their SHA256 values. Original audit code is in [tools](tools/).
