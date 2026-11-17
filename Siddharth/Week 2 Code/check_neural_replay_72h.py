"""Regression check: serialized inference uses identical PyTorch operations.

No training and no market data. Includes large cancelling contributions,
near-zero outputs, multiple architectures, missing inputs and partial batches.
"""
import tempfile
from pathlib import Path
import joblib
import numpy as np
import torch
from torch import nn
import fit_neural_excursions_72h as s


def main():
    rng = np.random.default_rng(48)
    torch.manual_seed(48)
    torch.set_num_threads(s.THREADS)
    raw = rng.normal(size=(137, 77))
    raw[0, :4] = np.nan
    prep = s.base.transform_fit(raw)
    values = np.clip(s.base.transform(raw, prep), -8, 8).astype(np.float32)
    for hidden in [[32], [32, 16], [64, 32]]:
        layers = []
        dims = [154]+hidden+[1]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        model = nn.Sequential(*layers).eval()
        with torch.no_grad():
            model[-1].weight.mul_(1000.)
            # Force output near zero at a nontrivial input, exercising cancellation.
            model[-1].bias.sub_(model(torch.from_numpy(values[:1])).ravel()[0])
        weights = [(layer.weight.detach().numpy().copy(), layer.bias.detach().numpy().copy())
                   for layer in model if isinstance(layer, nn.Linear)]
        bundle = dict(prep=prep, weights=weights, scale=300., config=dict(hidden=hidden))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'weights.joblib'
            joblib.dump(bundle, path)
            replay = joblib.load(path)
            for n in [1, 17, 64, 137]:
                with torch.no_grad():
                    expected = np.maximum(0., model(torch.from_numpy(values[:n])).numpy().ravel().astype(float)*300.)
                np.testing.assert_array_equal(s.predict_candidate(replay, raw[:n]), expected)
    print('PASS: exact PyTorch/serialized-array inference, cancellation, near-zero outputs, missing values, three architectures and four batch sizes; no fitting')


if __name__ == '__main__':
    main()
