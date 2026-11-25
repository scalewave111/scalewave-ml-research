# Separate model interfaces

These interfaces load the existing frozen weights. They do not reconstruct
features, estimate historical scales, fit calibration, or train models.
Your team supplies the prepared feature table described in FEATURES.md.

```text
SAK_GROSS_V01/
  SAK_DIR_V01.py
  SAK_MAG_V01.py
  SAK_GROSS_V01.py
  FEATURES.md
  requirements.txt
  model_files/
```

Keep all three Python files together: the magnitude and combined interfaces
import shared loading/input-validation utilities from the direction interface.
No imports from the older model_handoffs directory are needed.

```python
from SAK_DIR_V01 import DirectionModel
from SAK_MAG_V01 import MagnitudeModel
from SAK_GROSS_V01 import GrossModel

direction = DirectionModel().predict(prepared_features)
magnitude = MagnitudeModel().predict(prepared_features)
combined = GrossModel().predict(prepared_features)
```

The default artifact directory is model_files beside these scripts. An external
artifact path can be supplied as the constructor argument or --artifact-dir.

```powershell
python SAK_DIR_V01.py --input features.parquet --output direction.csv
python SAK_MAG_V01.py --input features.parquet --output magnitude.csv
python SAK_GROSS_V01.py --input features.parquet --output combined.csv
```

Input may be CSV or Parquet; existing output files are never overwritten.
Supply a sorted, unique, timezone-aware UTC minute index/column named
forecast_timestamp_utc. Direction needs146 features. Magnitude needs98 features
and a positive finite magnitude_scale_bps column supplied by your feature pipeline.
Joint prediction needs their147-feature union plus that scale. NaNs are supported;
infinite features, missing required columns and invalid timestamps are rejected.

The model files cover2023Q2--2024Q4 and are chosen automatically by timestamp.
Other dates are rejected. The signed score and fixed policy remain:
score=(2*p_up-1)*magnitude_bps; eligible=abs(score)>=10.
Entry is forecast+1minute; exit is forecast+16minutes.

The frozen weights remain .cbm and .txt model files, with calibration JSON.
The .py files are loading/prediction interfaces, not conversions of those weights
into Python source. Keep binary artifacts out of Git and use the agreed artifact
delivery route. No feature pipeline is run by these interfaces.
