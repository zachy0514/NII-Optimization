## NII-Optimization

This repository contains a Python implementation of a Net Interest Income (NII) optimization model designed for GSIB-style balance sheet planning. The model selects an asset mix and funding/liability mix subject to regulatory constraints such as TLAC, NSFR, and LCR while optimizing a mean-variance style objective.

Key features
- In-memory synthetic data generation for quick testing and development
- Constraint handling for NSFR, TLAC, LCR and bespoke balance-sheet rules
- Class-based, testable structure in `src/optimization.py`

Quick start
1. Create and activate the virtual environment (optional but recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. Run the optimizer (uses synthetic data in-memory):

```powershell
.\.venv\Scripts\python.exe src\optimization.py
```

Files of interest
- `src/optimization.py` — main optimizer and data generator (class-based)
- `input/` — place to add your own covariance or rate files if needed (not required)
- `pyproject.toml` / `requirements.txt` — project dependencies

Notes and troubleshooting
- The code is designed to run without reading or writing Excel files by default; it generates synthetic data in-memory. If you want to load your own data, add Excel files to the `input/` folder and adapt `DataGenerator.load_data` / `load_cov`.
- If you see an error about missing packages (e.g., `openpyxl`), install them via pip in the virtualenv.

Next steps
- Add unit tests for the constraint functions and the optimizer result validation.
- Parameterize the optimizer via a CLI or small config file.

License
MIT
