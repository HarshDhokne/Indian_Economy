# Indian Economy Sector Dashboard

A Streamlit dashboard for exploring quarterly Indian economy indicators across sectors. It includes year and sector filters, time-series and sector-comparison charts, filtered CSV export, and a simple imports estimate using the saved linear regression model.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run economy_model.py
```

The dashboard expects `Indian_Economy_Sectorwise_Dataset.csv` and `economy.pkl` in the project directory. `Untitled2 (1).py` contains the model-training workflow.

## Model note

The saved regression uses year alone to predict row-level `Imports_Crore`. Its estimate is not a national imports forecast, and it does not model sector or quarter effects.