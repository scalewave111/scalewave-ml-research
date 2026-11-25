# 48-hour magnitude research

Predict the size of BTCUSDT endpoint price changes over 48 hours, ignoring direction. This project is separate from the 15-minute model integration.

The retained development model has MAE **180.798518 log bps**, compared with **182.036977** for the original baseline on 18,481 rows. This is development evidence, not proof of trading profit.

| Folder | Contents |
| --- | --- |
| [feature_exploration](feature_exploration/README.md) | Earlier feature investigations, notebooks, reports and synthetic verification |
| [playground](playground/README.md) | Research code, experiment history and retained model artifacts |
| [reports](reports/research_summary.md) | Summary, numerical results and existing handoff reports |
| [results](results/README.md) | Magnitude experiment registry and milestone scores |
| [docs](docs/reproducibility.md) | Replay instructions, targets, limitations and data access |
| [environments](environments/magnitude/README.md) | Magnitude dependencies |
| [tools](tools/replay_magnitude.py) | Replay, package validation and summary generation |
| [artifacts](artifacts/README.md) | Current file hashes and original source provenance |

The playground's internal structure and fitted models were preserved. Original data remains external. Historical scripts can train or change models; use the documented replay command for verification.

Read [next steps](CONTINUE_RESEARCH.md) before planning new research.
