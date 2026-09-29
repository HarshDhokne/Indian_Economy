from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Indian_Economy_Sectorwise_Dataset.csv"
MODEL_PATH = BASE_DIR / "economy.pkl"
INDICATORS = {
	"GDP (lakh crore)": "GDP_Lakh_Crore",
	"Growth (%)": "Growth_%",
	"Employment (million)": "Employment_Million",
	"Exports (crore)": "Exports_Crore",
	"Imports (crore)": "Imports_Crore",
	"FDI (crore)": "FDI_Crore",
	"Inflation (%)": "Inflation_%",
	"CPI": "CPI",
	"Industrial production index": "IIP",
	"Tax revenue (crore)": "Tax_Revenue_Crore",
}


st.set_page_config(page_title="Indian Economy | Sector Dashboard", layout="wide")


@st.cache_data
def load_data():
	data = pd.read_csv(DATA_PATH)
	data["Period"] = data["Year"].astype(str) + " " + data["Quarter"]
	return data


@st.cache_resource
def load_model():
	return joblib.load(MODEL_PATH)


economy = load_data()
years = sorted(economy["Year"].unique())
sectors = sorted(economy["Sector"].unique())

st.title("Indian Economy")
st.caption("Quarterly indicators across sectors, with a simple imports trend estimate.")

with st.sidebar:
	st.header("Explore")
	selected_sectors = st.multiselect("Sectors", sectors, default=sectors)
	year_range = st.slider(
		"Years",
		min_value=int(years[0]),
		max_value=int(years[-1]),
		value=(int(years[0]), int(years[-1])),
	)
	indicator_label = st.selectbox("Indicator", list(INDICATORS))
	aggregation = st.radio("Combine sectors", ["Average", "Total"], horizontal=True)

filtered = economy[
	economy["Sector"].isin(selected_sectors)
	& economy["Year"].between(year_range[0], year_range[1])
]
indicator = INDICATORS[indicator_label]
agg_func = "sum" if aggregation == "Total" else "mean"

if filtered.empty:
	st.info("Select at least one sector to see the economy data.")
	st.stop()

latest = filtered.sort_values(["Year", "Quarter"]).groupby(["Year", "Quarter"], sort=False).tail(1)
latest_period = latest.iloc[-1]["Period"]
latest_rows = filtered[filtered["Period"] == latest_period]
metric_cols = st.columns(4)
metric_cols[0].metric("Latest period", latest_period)
metric_cols[1].metric("Average GDP", f"{latest_rows['GDP_Lakh_Crore'].mean():,.1f} lakh cr")
metric_cols[2].metric("Average growth", f"{latest_rows['Growth_%'].mean():.2f}%")
metric_cols[3].metric("Imports", f"{latest_rows['Imports_Crore'].sum():,.0f} cr")

overview_tab, sectors_tab, model_tab, data_tab = st.tabs(
	["Overview", "Sector comparison", "Imports estimate", "Data"]
)

with overview_tab:
	st.subheader(f"{indicator_label} over time")
	quarterly = filtered.groupby(["Year", "Quarter", "Period"], as_index=False)[indicator].agg(agg_func)
	quarterly = quarterly.sort_values(["Year", "Quarter"])
	st.line_chart(quarterly.set_index("Period")[[indicator]], y_label=indicator_label)
	st.caption(f"Quarterly {aggregation.lower()} across the selected sectors.")

with sectors_tab:
	st.subheader(f"Sector comparison · {year_range[1]}")
	comparison = economy[
		(economy["Year"] == year_range[1]) & economy["Sector"].isin(selected_sectors)
	].groupby("Sector")[indicator].agg(agg_func).sort_values(ascending=False)
	st.bar_chart(comparison, horizontal=True, x_label=indicator_label)

with model_tab:
	st.subheader("Imports estimate by year")
	st.write(
		"This is the saved linear regression trained on Year and row-level Imports_Crore. "
		"It estimates an individual sector-quarter observation, not total national imports."
	)
	forecast_year = st.number_input(
		"Estimate for year",
		min_value=int(years[0]),
		max_value=int(years[-1]) + 25,
		value=int(years[-1]) + 5,
		step=1,
	)
	try:
		model = load_model()
		estimate = float(model.predict([[int(forecast_year)]])[0][0])
		st.metric(f"Estimated imports in {int(forecast_year)}", f"{estimate:,.0f} crore")
		annual_imports = economy.groupby("Year")["Imports_Crore"].mean().rename("Observed average")
		forecast_point = pd.Series({int(forecast_year): estimate}, name="Model estimate")
		chart_data = pd.concat([annual_imports, forecast_point], axis=1).sort_index()
		st.line_chart(chart_data, y_label="Imports (crore)")
	except (FileNotFoundError, OSError, ValueError, AttributeError) as error:
		st.error(f"Could not load or run economy.pkl: {error}")

with data_tab:
	st.subheader("Filtered observations")
	st.dataframe(filtered.drop(columns="Period"), use_container_width=True, hide_index=True)
	st.download_button(
		"Download filtered CSV",
		data=filtered.drop(columns="Period").to_csv(index=False).encode("utf-8"),
		file_name="indian_economy_filtered.csv",
		mime="text/csv",
	)