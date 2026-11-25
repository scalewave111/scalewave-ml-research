# Set up on the future cloud host

These are instructions for Tariq or the TL. **No cloud resources have been created and no deployment has run.** A virtual machine runs the Python pipeline separately from the MLflow application, even if they share a machine.

## 1. Prepare access and choose the host

Use a company-approved account, operating system and persistent storage. Confirm access to the private GitHub repository, the dataset and the configured MLflow URL. If MLflow stays on the office PC, the cloud host needs an approved connection to it, and artifact retrieval depends on its availability.

Provide Python **3.12.5**, Git, enough memory/storage and private dashboard access. Do not copy another PC's virtual environments. The pinned dashboard dependency lock came from Windows; verify installation on the selected cloud OS. No specific cloud OS has been tested here.

## 2. Get the approved source commit

Clone or update `scalewave111/scalewave-ml-research` using company GitHub authentication. Select the reviewed integration commit (`trq` until merged). Record `git rev-parse HEAD` in the deployment record.

From the repository root, make the original package source commit available:

```shell
git fetch origin 0bb280de7c924da0d8cc96042556eb42e25f6e38
cd Tariq/model_integration
```

No branch switch is required to read the pinned source commit. The setup script uses `git show` and checks the exact file bytes against `config/models.json`.

## 3. Retrieve the package documents on the host

From `Tariq/model_integration`, using Python 3.12.5:

```shell
python scripts/prepare_packages.py --source-repo ../..
```

This restores only the selected package documents to ignored `runtime/packages/v002/` folders. It does not train or execute the models. Conflicting files fail validation instead of being overwritten.

## 4. Download trained artifacts from MLflow

Reuse an existing MLflow client environment, or create one on the host. The following commands assume `python` is the approved Python 3.12.5 executable.

**Linux shell:**

```shell
python -m venv .venv-mlflow
.venv-mlflow/bin/python -m pip install --no-cache-dir mlflow-skinny==3.18.0
.venv-mlflow/bin/python scripts/download_models.py
```

**Windows PowerShell:**

```powershell
python -m venv .venv-mlflow
& '.\.venv-mlflow\Scripts\python.exe' -m pip install --no-cache-dir mlflow-skinny==3.18.0
& '.\.venv-mlflow\Scripts\python.exe' scripts/download_models.py
```

Enter credentials at the local prompts. Never commit them. Downloads are checked against the pinned manifests and stored under `runtime/artifacts/v002/`. The script does not alter MLflow records.

## 5. Install model dependencies and verify

**Linux shell:**

```shell
python -m venv .venv-model
.venv-model/bin/python -m pip install --no-cache-dir -r runtime/packages/v002/direction/requirements.txt
.venv-model/bin/python -B -m unittest discover -s tests -v
.venv-model/bin/python -B pipeline/run.py --mode replay --model-python .venv-model/bin/python
```

**Windows PowerShell:**

```powershell
python -m venv .venv-model
& '.\.venv-model\Scripts\python.exe' -m pip install --no-cache-dir -r runtime/packages/v002/direction/requirements.txt
& '.\.venv-model\Scripts\python.exe' -B -m unittest discover -s tests -v
& '.\.venv-model\Scripts\python.exe' -B pipeline/run.py --mode replay --model-python '.\.venv-model\Scripts\python.exe'
```

The two selected packages declare identical dependencies. Expect `REPLAY_PASSED_DEVELOPMENT_ONLY`, target alignment `MATCH`, trading disabled and approximately +0.11432498591035122 bps. A mismatch must be investigated; do not change expected values to force a pass. This tests synthetic examples, not live feature correctness or strategy performance.

## 6. Preview the dashboard and arrange private access

```shell
python dashboard/launch.py
```

Open `http://127.0.0.1:8501` on the host. For a remote browser, the TL must arrange authenticated private access or an approved tunnel. The launcher does not expose a public service. Restart it after code changes.

## 7. Continue implementation

Connect the approved dataset using [DATA_READINESS.txt](DATA_READINESS.txt). Build real-data preparation before scheduling predictions. Do not schedule the example replay and call it a market pipeline.

Then define scheduling, startup/restart behavior, alerts, retention, backups, trading rules, risk controls and paper execution. Confirm MLflow database and artifact persistence separately. Live trading needs its own tested execution and operational readiness.

## Updates

Deploy a reviewed Git commit, keep matching configuration/model versions, reinstall only changed dependencies, and verify replay before acceptance. Reuse environments and matching files. Keep the incumbent until a replacement passes. Do not delete remote model history or source data during routine cleanup.
