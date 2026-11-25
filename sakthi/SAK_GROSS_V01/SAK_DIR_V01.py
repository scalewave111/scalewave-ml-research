"""Frozen direction inference. The caller supplies all prepared features.

python SAK_DIR_V01.py --input features.parquet --output direction.csv
No feature construction, calibration fitting, or training occurs here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit, logit

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / 'model_files'


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class _FrozenBundle:
    def __init__(self, artifact_dir=None):
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else DEFAULT_ARTIFACT_DIR
        hashes = json.loads((self.artifact_dir / 'artifact_hashes.json').read_text())
        for name, expected in hashes.items():
            if Path(name).name != name or _sha256(self.artifact_dir / name) != expected:
                raise ValueError('Artifact integrity check failed: ' + name)
        self.bundle = json.loads((self.artifact_dir / 'bundle.json').read_text())
        self.schema = json.loads((self.artifact_dir / 'feature_schema.json').read_text())
        if self.bundle['model_version'] != 'SAK_GROSS_V01' or self.bundle['threshold_bps'] != 10:
            raise ValueError('Unexpected model identity or threshold.')
        self._models = {}

    def _prepare(self, frame, kind):
        frame = frame.copy()
        if 'forecast_timestamp_utc' in frame:
            frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop('forecast_timestamp_utc'), errors='raise'))
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError('A timezone-aware forecast_timestamp_utc column or DatetimeIndex is required.')
        frame.index = frame.index.tz_convert('UTC')
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError('Timestamps must be sorted and unique.')
        if not frame.index.equals(frame.index.floor('min')):
            raise ValueError('Timestamps must be exact minute boundaries.')
        missing = set(self.schema[kind]) - set(frame.columns)
        if missing:
            raise ValueError('Missing prepared features: ' + ', '.join(sorted(missing)))
        values = frame[self.schema[kind]].to_numpy(dtype='float32')
        if np.isinf(values).any():
            raise ValueError('Infinity is invalid; use NaN for unavailable features.')
        output = pd.DataFrame(index=frame.index)
        output.index.name = 'forecast_timestamp_utc'
        output['entry_timestamp_utc'] = output.index + pd.Timedelta(minutes=1)
        output['exit_timestamp_utc'] = output.index + pd.Timedelta(minutes=16)
        return frame, values, output

    def _folds(self, index):
        covered = np.zeros(len(index), dtype=bool)
        matches = []
        for fold in self.bundle['folds']:
            selected = (index >= pd.Timestamp(fold['valid_from'])) & (index < pd.Timestamp(fold['valid_until']))
            if selected.any():
                covered |= selected
                matches.append((fold, selected))
        if not covered.all():
            raise ValueError('Timestamp outside frozen coverage [2023-04-01, 2025-01-01).')
        return matches


class DirectionModel(_FrozenBundle):
    """Load SAK_DIR_V01; accept 146 prepared direction features."""
    def predict(self, frame):
        frame, values, output = self._prepare(frame, 'direction')
        output['p_up'] = np.nan
        for fold, selected in self._folds(frame.index):
            quarter = fold['quarter']
            if quarter not in self._models:
                model = CatBoostClassifier()
                model.load_model(str(self.artifact_dir / fold['direction_file']))
                calibration = json.loads((self.artifact_dir / fold['calibration_file']).read_text())
                self._models[quarter] = model, calibration
            model, calibration = self._models[quarter]
            raw = model.predict_proba(values[selected], thread_count=2)[:, 1]
            center = logit(np.clip(calibration['prior_up'], 1e-6, 1 - 1e-6))
            p = expit(center + calibration['lambda'] * (logit(np.clip(raw, 1e-6, 1 - 1e-6)) - center))
            output.loc[selected, 'p_up'] = p
            output.loc[selected, 'model_fold'] = quarter
        output['p_down'] = 1 - output.p_up
        output['direction_model'] = 'SAK_DIR_V01'
        return output


def _run_cli(model_class):
    parser = argparse.ArgumentParser(description=model_class.__doc__)
    parser.add_argument('--artifact-dir', default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument('--input', required=True, help='Prepared CSV or Parquet feature table.')
    parser.add_argument('--output', required=True, help='New CSV or Parquet output path.')
    args = parser.parse_args()
    source, destination = Path(args.input), Path(args.output)
    if destination.exists():
        raise ValueError('Output already exists; choose a new path.')
    if source.suffix.lower() not in ['.csv', '.parquet'] or destination.suffix.lower() not in ['.csv', '.parquet']:
        raise ValueError('Input and output must be CSV or Parquet.')
    frame = pd.read_parquet(source) if source.suffix.lower() == '.parquet' else pd.read_csv(source)
    result = model_class(args.artifact_dir).predict(frame)
    if destination.suffix.lower() == '.parquet':
        result.to_parquet(destination)
    else:
        result.to_csv(destination)


if __name__ == '__main__':
    _run_cli(DirectionModel)
