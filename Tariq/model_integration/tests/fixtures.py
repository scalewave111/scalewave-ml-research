"""Tiny generated contracts for offline tests; no real model or submission copies."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for directory in ('pipeline', 'dashboard', 'scripts'):
    sys.path.insert(0, str(ROOT / directory))
import run as p
import state


def prepare_fixture(test):
    parent = ROOT / '.verification'
    parent.mkdir(exist_ok=True)
    temp = tempfile.TemporaryDirectory(prefix='contract-', dir=parent)
    workspace = Path(temp.name).resolve()
    assert workspace.is_relative_to(parent.resolve())
    test.addCleanup(temp.cleanup)
    config = json.loads((ROOT / 'config/models.json').read_text())
    patcher = patch.object(p, 'ROOT', workspace)
    patcher.start()
    test.addCleanup(patcher.stop)
    for role, model in config['models'].items():
        package = workspace / model['package_dir']
        package.mkdir(parents=True)
        artifacts = workspace / model['artifact_dir']
        artifacts.mkdir(parents=True)
        artifact = artifacts / 'fixture.txt'
        artifact.write_text('Synthetic test artifact, not a trained model.')
        identity = {'symbol': config['symbol'], 'horizon': config['horizon'],
                    'forecast_time_utc': '2024-10-01T00:00:00Z'}
        manifest = {k: model[k] for k in ('target', 'training_cutoff_utc')}
        manifest.update(model_version=model['version'], symbol=config['symbol'], horizon=config['horizon'],
                        input_features=['test_feature'], output_units=model['units'], python_version='3.12.5',
                        artifacts=[{'file': artifact.name, 'sha256': p.digest(artifact)}])
        records = {
            'manifest.json': manifest,
            'example_input.json': dict(identity, features={'test_feature': 1.0}, scale_bps=12),
            'expected_output.json': dict(identity, model_version=model['version'],
                                        **{model['output_field']: .6 if role == 'direction' else 20.0}),
            'evaluation.json': {'evaluation_start_utc': '2023-04-01T00:00:00Z',
                                'evaluation_end_utc': '2024-12-31T00:00:00Z', 'row_count': 2},
        }
        for name, value in records.items():
            (package / name).write_text(json.dumps(value))
        (package / 'predict.py').write_text("INPUT_COLUMNS = ('test_feature',)\n")
        (package / 'requirements.txt').write_text('numpy==2.1.1\n')
        model['package_sha256'] = {file.name: p.digest(file) for file in package.iterdir()}
    (workspace / 'config').mkdir()
    (workspace / 'config/models.json').write_text(json.dumps(config))
    return config
