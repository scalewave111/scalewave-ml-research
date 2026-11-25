"""Frozen magnitude inference from prepared features and supplied scale.

python SAK_MAG_V01.py --input features.parquet --output magnitude.csv
The historical scale and features are not reconstructed here.
"""
import numpy as np
import lightgbm as lgb
from SAK_DIR_V01 import _FrozenBundle, _run_cli


class MagnitudeModel(_FrozenBundle):
    """Load SAK_MAG_V01; accept 98 prepared features and magnitude_scale_bps."""
    def predict(self, frame):
        frame, values, output = self._prepare(frame, 'magnitude')
        if 'magnitude_scale_bps' not in frame:
            raise ValueError('Supply the precomputed magnitude_scale_bps column.')
        scale = frame.magnitude_scale_bps.to_numpy(dtype=float)
        if not np.all(np.isfinite(scale) & (scale > 0)):
            raise ValueError('Magnitude scale must be finite and positive.')
        output['predicted_magnitude_bps'] = np.nan
        for fold, selected in self._folds(frame.index):
            quarter = fold['quarter']
            if quarter not in self._models:
                self._models[quarter] = lgb.Booster(model_file=str(self.artifact_dir / fold['magnitude_file']))
            normalized = self._models[quarter].predict(values[selected], num_threads=2)
            output.loc[selected, 'predicted_magnitude_bps'] = np.maximum(normalized, 0) * scale[selected]
            output.loc[selected, 'model_fold'] = quarter
        output['magnitude_model'] = 'SAK_MAG_V01'
        return output


if __name__ == '__main__':
    _run_cli(MagnitudeModel)
