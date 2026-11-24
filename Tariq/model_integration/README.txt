SCALEWAVE MODEL INTEGRATION - DEVELOPMENT HANDOFF

Current scope
-------------
This is a verified example-replay pipeline, dashboard and step monitor.
It is not yet a scheduled market-data or trading system. Trading stays disabled.
Start with DEPLOYMENT.txt for a fresh installation. DATA_READINESS.txt explains
what is needed before connecting historical market data.

Completed
---------
- Version-pinned BTCUSDT 15-minute direction and magnitude v002 packages.
- MLflow artifact retrieval with SHA-256 checks; matching files are reused.
- Input, output, dependency, timestamp and target-window validation.
- Replay of the supplied synthetic examples using fitted models.
- Signed score = (2 * p_up - 1) * net_magnitude_bps.
- Streamlit dashboard with Plotly charts, checks, history and score simulator.
- Monitoring of alignment, files, inputs, direction, magnitude, combine and save.
  Each step records status, time and errors. Heartbeat every 5 seconds; dashboard
  refreshes monitoring every 3 seconds. A running heartbeat older than 20 seconds
  is labelled possibly interrupted, not conclusively dead.

Selected models
---------------
Direction: sid_direction-15m-v002
MLflow run: 5b431d95a1e447ff8b4aabcd6e27d546
Magnitude: sid_net-magnitude-15m-v002
MLflow run: 8317954297ce44059067e1bac89afeb9
Experiment: BTC_15M_TEAM
Handoff commit: 588c6918c6d2ccb0444fc0bb3f6e1060ebb7d227
Package source: 0bb280de7c924da0d8cc96042556eb42e25f6e38
Both targets: t+1 to t+16 minutes; whole-minute issuance is permitted.
Training cutoff: 2024-10-01 UTC. Original training commits are unavailable.
Matching evaluation dates/counts are checked; row-level identity and historical
scores have not been independently reproduced. Magnitude predicts a conditional
median of absolute simple returns in bps. The combined score is not an established
conditional expected return.

Verified example (synthetic inputs, not a market signal)
-------------------------------------------------------
p_up: 0.5071603369109142
net_magnitude_bps: 7.983212754702233
predicted_signed_score_bps: 0.11432498591035122
Replay tolerance: absolute 1e-8, relative zero.

Storage and version control
---------------------------
GitHub holds code, configuration, small package documents and dependency records.
MLflow holds trained artifacts; config.json pins exact run IDs and package hashes.
Do not commit credentials, datasets, trained binaries or virtual environments.
The active artifacts live in artifacts/v002/direction and artifacts/v002/magnitude.
Results replace results/current; each history keeps at most 100 compact summaries.
Tests reuse .verification. No additional model copies are required for publishing.
The local v001 cleanup was completed previously; remote history was preserved.
upgrade_v002.py is a historical migration helper, not the fresh-install procedure.

How to run after setup (PowerShell, from this folder)
---------------------------------------------------
& '.\.venv-model\Scripts\python.exe' -B pipeline.py --mode check
& '.\.venv-model\Scripts\python.exe' -B pipeline.py --mode replay --model-python '.\.venv-model\Scripts\python.exe'
& '.\.venv-model\Scripts\python.exe' -B -m unittest -v test_pipeline test_monitoring
& '.\.venv-model\Scripts\python.exe' -B -m unittest discover -s dashboard -p test_dashboard.py
& '.\.venv-model\Scripts\python.exe' dashboard/launch.py

Modes and results
-----------------
check: validates files; does not execute models.
reported-example: formula demonstration using supplied expected outputs only.
replay: executes the frozen models and compares predictions with supplied examples.
REPLAY_PASSED_DEVELOPMENT_ONLY means the package replay passed, not a backtest.
FAILED or INTERRUPTED means downstream consumers must not use partial predictions.
The dashboard withholds outputs when results/configuration/monitoring disagree.
Concurrent CLI runs are prevented with results/runner.lock. After a crash, confirm
that the process has stopped before removing a leftover lock.

Still to do
-----------
1. Connect the owner's market data and verify causal feature preparation.
2. Evaluate combined scores against actual t+1 to t+16 signed returns.
3. Implement trading policy, costs and risk limits; the +/-10 bps filter is only
   a simulator proposal, not an approved trading rule.
4. Add scheduling, fresh-data checks, external alerts and recovery procedures.
5. Implement and test paper execution before considering live trading.
6. Deploy on an agreed host and configure private access and operations.
CI/CD is an optional future addition; this handoff does not install it.
