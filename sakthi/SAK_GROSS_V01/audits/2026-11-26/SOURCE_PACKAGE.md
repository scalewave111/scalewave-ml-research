# Source package and replay inputs

The audited source is `SAK_GROSS_V01_COMPLETE_20261126.zip`.

- ZIP SHA256: `bd0c6a1ddb2a836dc17d843a3e4247346b6dd2e7b8fb80996cba40366e526ed9`.
- Original interface reference: commit `3acaf171746ff175ceb5e525f05cca3cc89263bb` in this repository.
- Full ZIP inventory: [ARCHIVE_RECEIPT.json](ARCHIVE_RECEIPT.json).
- Recorded input reconstruction: [INPUT_RECONSTRUCTION.json](INPUT_RECONSTRUCTION.json).
- Runtime versions: [RUNTIME_CHECK.json](RUNTIME_CHECK.json).

The source ZIP, fitted weights, H4 data, rebuilt matrices, replay partitions and isolated Python installation remain in the local research handoff. This GitHub folder contains the audit reports, small evidence files, trade ledgers and the exact helper scripts used. Local paths in receipts record the original execution environment.

For a full rerun, obtain the matching ZIP through the team's handoff, verify its SHA256 and extract its `SAK_GROSS_V01_COMPLETE` directory under this folder's `source/`. Restore access to the H4 native release, the H4 arena cache and the inherited APEX decision calendar. The reconstruction and native-probe scripts contain the original Windows paths; map those to the corresponding unchanged inputs on another machine.

Use the supplied package's dependency pins, including **LightGBM 4.7.0 for retraining**. The audit separately records the 4.6.0 mismatch and the controlled 4.7.0 pass. Only Q4 was independently retrained; all seven frozen quarters were replayed.

With the source package restored, `python tools/analyse_ledger.py` recomputes all-entry, monthly, quarterly and daily accounting from its replay partitions. `python tools/write_findings.py` regenerates the report from the saved audit receipts. Run helpers in a separate working copy because they write outputs alongside their inputs.

Handoff 4a is evaluation-only and was not accessed by this audit. These helpers do not authorize a new model search or H4a evaluation.
