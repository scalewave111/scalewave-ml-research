"""Combined frozen prediction at +/-10 bps. All features are caller-supplied.

python SAK_GROSS_V01.py --input features.parquet --output predictions.csv
"""
import numpy as np
from SAK_DIR_V01 import DirectionModel, _run_cli
from SAK_MAG_V01 import MagnitudeModel


class GrossModel:
    """Accept the union of prepared features plus magnitude_scale_bps."""
    def __init__(self, artifact_dir=None):
        self.direction = DirectionModel(artifact_dir)
        self.magnitude = MagnitudeModel(artifact_dir)

    def predict(self, frame):
        direction = self.direction.predict(frame)
        magnitude = self.magnitude.predict(frame)
        if not direction.index.equals(magnitude.index) or not direction.model_fold.equals(magnitude.model_fold):
            raise ValueError('Direction and magnitude predictions are not aligned.')
        result = direction.copy()
        result['predicted_magnitude_bps'] = magnitude.predicted_magnitude_bps
        result['magnitude_model'] = magnitude.magnitude_model
        result['signed_score_bps'] = (2 * result.p_up - 1) * result.predicted_magnitude_bps
        result['eligible'] = result.signed_score_bps.abs().ge(10)
        result['predicted_side'] = np.sign(result.signed_score_bps).astype(int)
        result['trade_side'] = np.where(result.eligible, result.predicted_side, 0)
        return result


if __name__ == '__main__':
    _run_cli(GrossModel)
