"""Read-only MLflow retrieval of manifest-listed artifacts; never logs or deploys."""
import getpass
import argparse
import os
from pathlib import Path
import shutil

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
from run import ROOT, read, inside, digest, inspect_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/models.json')
    args = parser.parse_args()
    config = read(args.config)
    username = os.environ.get('MLFLOW_TRACKING_USERNAME') or input('MLflow username: ')
    password = os.environ.get('MLFLOW_TRACKING_PASSWORD') or getpass.getpass('MLflow password (hidden): ')
    os.environ['MLFLOW_TRACKING_USERNAME'] = username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = password
    os.environ['MLFLOW_DISABLE_AGENT_HINT'] = '1'
    try:
        from mlflow import MlflowClient
        client = MlflowClient(tracking_uri=config['tracking_uri'])
        experiment = client.get_experiment_by_name(config['experiment'])
        if experiment is None:
            raise ValueError('Configured experiment not found')
        for role, model in config['models'].items():
            checks = inspect_package(config, role)
            if any(c['status'] == 'FAIL' and not c['check'].startswith('artifact:') for c in checks):
                raise ValueError(role + ': pinned package checks failed before retrieval')
            run = client.get_run(model['run_id'])
            if str(run.info.experiment_id) != str(experiment.experiment_id):
                raise ValueError(role + ': run belongs to a different experiment')
            manifest = read(inside(ROOT, model['package_dir']) / 'manifest.json')
            files = []
            queue = ['']
            while queue:
                prefix = queue.pop()
                for entry in client.list_artifacts(model['run_id'], prefix):
                    if entry.is_dir:
                        queue.append(entry.path)
                    else:
                        files.append(entry.path)
                if len(files) + len(queue) > 1000:
                    raise ValueError('Unexpectedly large artifact listing; review run manually')
            destination = inside(ROOT, model['artifact_dir'])
            destination.mkdir(parents=True, exist_ok=True)
            # Reuse one staging directory for this model version across retries.
            staging = ROOT / 'runtime/.downloads'
            staging.mkdir(parents=True, exist_ok=True)
            stage = inside(staging, model['version'])
            stage.mkdir(parents=True, exist_ok=True)
            for entry in manifest['artifacts']:
                final = inside(destination, entry['file'])
                if final.exists():
                    if digest(final) != entry['sha256']:
                        raise ValueError('Existing artifact hash mismatch: ' + final.name)
                    print(role, final.name, 'already verified')
                    continue
                candidates = [p for p in files if Path(p).name == entry['file']]
                if len(candidates) != 1:
                    raise ValueError('Expected exactly one remote artifact named ' + entry['file'])
                downloaded = Path(client.download_artifacts(model['run_id'], candidates[0], str(stage))).resolve()
                if not downloaded.is_relative_to(stage.resolve()):
                    raise ValueError('Download returned an unexpected local destination')
                if digest(downloaded) != entry['sha256']:
                    raise ValueError('Downloaded file differs from pinned manifest: ' + entry['file'])
                final.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(downloaded), str(final))
                print(role, final.name, 'downloaded and hash verified')
        print('Retrieval complete. Run pipeline/run.py --mode check, then replay in the documented environment.')
    finally:
        os.environ.pop('MLFLOW_TRACKING_PASSWORD', None)
        os.environ.pop('MLFLOW_TRACKING_USERNAME', None)


if __name__ == '__main__':
    main()
