# Magnitude environment

Python 3.10.0 and the direct package versions in requirements.txt were present during this handoff. These are tested local direct dependencies, not a complete transitive lockfile or a guarantee for a new OS. Create a separate virtual environment inside this repository. Do not install globally.

Magnitude inference passed in this environment. Original playground BUILD_ENVIRONMENT.json describes a different earlier Python 3.12 frozen-baseline package; it is historical, not the current forest replay requirement.

The historical training scripts may use optional libraries such as XGBoost or PyTorch. Those are not required for the current retained magnitude inference. Their old versions/settings remain in the linked experiment environment receipts. Retraining every old experiment is not a validated capability of this handoff.

PDF regeneration uses PyMuPDF 1.28.2; it is optional for readers. The earlier feature synthetic test also requires matplotlib 3.10.9 and PyArrow. See [reproducibility](../../docs/reproducibility.md).
