SCALEWAVE LOCAL DASHBOARD

Start from PowerShell (any current folder):
& 'C:\ScaleWave\Tools\Python 3125\python.exe' 'C:\ScaleWave\Model_Integration\dashboard\launch.py'

The first launch installs Streamlit and Plotly into ONE .venv-dashboard environment.
This is separate from .venv-model. No trained files or datasets are copied.
Successful setup records exact package versions in requirements.lock.txt for reuse.
The launcher reuses the installed environment on later runs. pip uses no download cache.

Open http://127.0.0.1:8501 in a browser on this PC.
Keep PowerShell running while using the page. Press Ctrl+C to stop it.
This is a local application, not a standalone HTML file. No public hosting is enabled.
If port 8501 is occupied, stop your earlier dashboard process before starting another.

Views:
- Overview: saved example outputs and target timeline, plus project readiness.
- Models & evidence: selected versions, MLflow links and owner-submitted evaluations.
- Checks & history: verification details, compact historical attempts and current JSON.
- Score simulator: hypothetical inputs; never runs models or changes trading settings.
- Pipeline monitoring: step status, UTC start/end, duration, error details and heartbeat.
  This panel refreshes every 3 seconds; the running process writes a heartbeat every
  5 seconds. A running record older than 20 seconds is labelled possibly interrupted.
  It does not prove process termination. Inspect the terminal before clearing a lock.

Refresh saved records rereads integration JSON files. There is no automatic live feed.
Recorded checks are historical evidence. Monitoring observes local process progress;
it does not establish live-data freshness or remote-service availability.
The page withholds current prediction metrics if the configuration changed or the
latest result failed. Owner-submitted evaluation scores are not independent backtests.
The 10-bps filter is proposed; live features, execution and risk controls are unfinished.
Original training commits are unavailable. Historical row-level identity is unverified.

Storage:
Only application code, one environment and a small dependency lock are added.
The page reads existing results/current and results/history.json without duplicating
them. It does not load model binaries, contact MLflow automatically, or save simulations.
monitor.json is updated atomically in results/current; results/monitor_history.json
retains at most 100 compact attempt summaries. No files accumulate per heartbeat.
The explicit Download JSON button saves only if you choose it in the browser.

Move to company PC:
Transfer application code through GitHub with the integration code; recreate the
environment from requirements.lock.txt and place records in the same relative layout.
Do not copy this PC's virtual environment. Keep localhost binding until company access
controls and a hosting arrangement are established.

Verification:
python -B -m unittest discover -s dashboard -p test_dashboard.py
These are offline data/logic tests. They do not replace checking browser rendering.
