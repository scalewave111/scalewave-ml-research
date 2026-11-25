# Operations dashboard

Run `python dashboard/launch.py` from the project root. The launcher reuses one `.venv-dashboard` environment and serves on `http://127.0.0.1:8501` on the same machine.

| View | Purpose |
| --- | --- |
| Overview | Selected versions, available verified results and readiness |
| Models & evidence | Model references and evaluation reports when retrieved on the host |
| Checks & history | Validation results and compact attempt history |
| Score simulator | Hypothetical direction/magnitude inputs; does not run models or place trades |
| Pipeline monitoring | Step status, duration, errors and heartbeat |

The dashboard reads `config/models.json` and saved results. It does not download or execute models. Missing package reports show as unavailable. Historical results that do not match the selected configuration are not displayed as current predictions.

Keep the terminal running; Ctrl+C stops the dashboard. Restart it after code updates. Cloud access requires an approved private access arrangement; the launcher stays bound to localhost. See [deployment](../docs/DEPLOYMENT.md).
