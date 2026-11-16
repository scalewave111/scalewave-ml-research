"""Synthetic diagnostic arithmetic checks; no real-data analysis or fitting."""
import numpy as np
import pandas as pd
import diagnose_excursions_72h as d


def main():
    frame = pd.DataFrame(dict(actual=[0., 10., 20.], training_median=[10., 10., 10.],
                              selected=[5., 12., 10.]))
    result = d.describe(frame, 'selected')
    np.testing.assert_allclose(result['mae'], 17/3)
    np.testing.assert_allclose(result['mae_gain_vs_median'], 1.)
    np.testing.assert_allclose(result['total_absolute_error_gain'], 3.)
    for key in ['fraction_helped', 'fraction_hurt', 'fraction_tied']:
        np.testing.assert_allclose(result[key], 1/3)
    np.testing.assert_allclose(result['adjustment_toward_actual_fraction'], .5)
    constant = d.describe(frame, 'training_median')
    assert constant['fraction_tied'] == 1 and constant['mean_abs_adjustment'] == 0
    assert constant['adjustment_toward_actual_fraction'] is None
    assert constant['ic'] is None
    assert d.bins([np.nan, 0, 1, 2, 3], 1, 2, ['low', 'mid', 'high']).tolist() == ['missing', 'low', 'low', 'mid', 'high']
    assert d.bins([1, 2], 1, 1, ['low', 'mid', 'high']).tolist() == ['low', 'high']
    # Disjoint-group total gains must sum to the full error decomposition.
    assert sum(d.describe(frame.iloc[[i]], 'selected')['total_absolute_error_gain'] for i in range(3)) == 3
    try:
        d.no_fitting()
    except RuntimeError:
        pass
    else:
        raise AssertionError('No-fitting guard failed')
    print('PASS: MAE/error decomposition, adjustment frequencies, constant baseline, missing/tied bins, no-fitting guard')


if __name__ == '__main__':
    main()
