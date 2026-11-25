# ScaleWave model integration

Connect the BTCUSDT 15-minute direction and magnitude models, inspect their combined signed score, and monitor each pipeline step.

**Status: development code prepared for a future cloud host. Nothing is deployed and trading is disabled.**

## What is included?

| Folder | Purpose |
| --- | --- |
| [pipeline](pipeline/run.py) | Model validation, example replay, signed-score calculation and step monitoring |
| [dashboard](dashboard/README.md) | Streamlit charts, results, monitoring and a score simulator |
| [config](config/models.json) | Exact model versions, MLflow runs, source commit and file hashes |
| [scripts](scripts/prepare_packages.py) | Retrieve pinned submission files and trained artifacts on the execution host |
| [tests](tests/) | Offline checks using tiny synthetic contracts; no trained models or company data |
| [docs](docs/DEPLOYMENT.md) | Cloud setup instructions and real-data requirements |

## Where are the models?

The original submission remains in the company GitHub repository, and trained artifacts remain in MLflow. This integration no longer includes copied submissions or trained model files.

On the future execution host, setup retrieves the exact pinned documents into `runtime/packages/` and binaries into `runtime/artifacts/`. That folder is ignored by Git. Existing verified files are reused. Uploading files to MLflow alone does not execute this pipeline.

| Selected model | Version | MLflow run |
| --- | --- | --- |
| Direction | `sid_direction-15m-v002` | `5b431d95a1e447ff8b4aabcd6e27d546` |
| Magnitude | `sid_net-magnitude-15m-v002` | `8317954297ce44059067e1bac89afeb9` |

Both target the price window from **t+1 to t+16 minutes**. Magnitude estimates a conditional median of absolute simple return in basis points.

## How it works

```text
Validate files and inputs → run both models → combine outputs → save results
                                      Monitoring records each step
                                      Dashboard displays the records
```

```text
signed_score_bps = (2 × probability_of_UP − 1) × predicted_magnitude_bps
```

This score is not an established expected return. The earlier fitted-model example replay passed with a score of approximately **+0.114325 bps**. That is historical packaging verification, not a live signal or backtest.

## What can I do now?

From this project folder, run the code-only tests:

```shell
python -B -m unittest discover -s tests -v
```

Use Python 3.12.5 for the eventual model runtime. Tests do not download models or contact MLflow.

To preview the dashboard using an approved Python installation:

```shell
python dashboard/launch.py
```

The launcher creates or reuses the dashboard environment. It serves only on the same machine at `http://127.0.0.1:8501`. Missing model evaluations and current predictions are shown as unavailable. Old results are historical and are withheld when their configuration no longer matches.

**Do not run setup/download commands on this PC if the intention is to keep all model files on the cloud host.** Follow [deployment instructions](docs/DEPLOYMENT.md) there instead.

## Completed and remaining work

| Completed | Still to do |
| --- | --- |
| Model identity, hash and contract checks | Connect real market data and verify feature preparation |
| Historical replay of supplied v002 examples | Evaluate combined scores against actual returns |
| Dashboard and step monitoring | Scheduling, stale-data checks and external alerts |
| Version references and host setup scripts | Trading rules, costs, risk limits and paper execution |
| Offline logic and failure-handling tests | Cloud deployment and later live-trading readiness |

The proposed ±10-bps filter is only a simulator rule. CI/CD is optional and is not installed. See [data readiness](docs/DATA_READINESS.txt) for the next functional milestone.

## Storage rules

Keep source and configuration in Git. Keep credentials, datasets, downloaded packages, trained artifacts and environments out of Git. Runtime results reuse `results/current`; compact histories are bounded. Test fixtures are temporary and removed after testing.
