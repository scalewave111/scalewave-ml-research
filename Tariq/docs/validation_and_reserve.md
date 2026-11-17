# Validation and the reserve

Validation means checking predictions on a later period than the training data. A reserve is data deliberately excluded from a particular research task.

| Rule | Magnitude | Timing |
| --- | --- | --- |
| Forecast cadence | 30 minutes | 30 minutes |
| Main evaluation | Jan 2024-Jan 22 2025; 18,481 rows | Apr 2023-Jun 2024; counts differ by horizon |
| Outer periods | Jan-Mar, Apr-Jun, Jul-Sep 2024; Oct 2024-Jan 22 2025 | Apr-Jun, Jul-Sep, Oct-Dec 2023; Jan-Mar, Apr-Jun 2024 |
| Training | Earlier expanding history; see original contracts | Earlier expanding history |
| Strict purge | Training label_end < validation start | Training label_end < validation start |
| Inner validation | Earlier three months for selection | Preceding three calendar months; strict purging |
| Preprocessing | Training data only | Training data only |

For timing, read research partitions April 2022-June 2024 only. Skip ALL July 2024 and later partitions, including the part of July before the reserve boundary. Research origins end before 2024-07-01T00:00:00Z. Research outcomes must end strictly before 2024-07-24T00:01:00Z.

Missing feature values do not justify dropping evaluation rows. Equal population counts alone do not independently establish equal row identities. The staged receipts support population and purge checks; raw row identities were not independently replayed locally.

The reserve is held out from the timing task. It is not historically unseen by the entire project: magnitude development used some dates within it. All results are development evidence, with repeated selection exposure.

F4Z publication, ingestion and normalization timing remain unresolved. Missing evidence is neither proof of leakage nor proof that the data is leakage-free.
