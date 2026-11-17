# Timing environment

Python 3.10.0 and the direct package versions in requirements.txt were present during this handoff. These are tested local direct dependencies, not a complete transitive lockfile or a guarantee for a new OS. Create a separate virtual environment inside this repository. Do not install globally.

Timing receipt verification uses the Python standard library. The requirements support local synthetic timing tests; they are not a reconstructed lock of the original Colab runtime. Colab supplies its authentication/Drive runtime separately. Do not launch cloud training to verify documentation.

The historical training scripts may use optional libraries such as XGBoost or PyTorch. Those are not required for the current retained magnitude inference or timing receipt verification. Their old versions/settings remain in the linked experiment environment receipts. Retraining every old experiment is not a validated capability of this handoff.

PDF regeneration uses PyMuPDF 1.28.2; it is optional for readers. The earlier feature synthetic test also requires matplotlib 3.10.9 and PyArrow. See [reproducibility](../../docs/reproducibility.md).
