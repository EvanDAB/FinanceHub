import requests
import pandas as pd
import datetime as dt
import os
import streamlit as st

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")

@st.cache_data(ttl=60*60)
def load_yield_curve_polygon(start=None, end=None):
    """
    Fetch CMT yields from Polygon's treasury-yields endpoint.
    Returns a tidy DataFrame with columns: date, maturity_years, yield
    """
    base = "https://api.polygon.io/fed/v1/treasury-yields"
    params = {
        "limit": 5000,
        "sort": "date.asc",
        "apiKey": POLYGON_API_KEY
    }
    if start: params["date.gte"] = pd.to_datetime(start).date().isoformat()
    if end:   params["date.lte"] = pd.to_datetime(end).date().isoformat()

    rows = []
    url = base
    while True:
        r = requests.get(url, params=params if url == base else None, timeout=30)
        r.raise_for_status()
        data = r.json()
        for item in data.get("results", []):
            d = pd.to_datetime(item["date"]).date()
            # Map Polygon fields to (years, label)
            mapping = {
                "yield_1_month": 1/12, "yield_3_month": 3/12, "yield_6_month": 6/12,
                "yield_1_year": 1, "yield_2_year": 2, "yield_3_year": 3,
                "yield_5_year": 5, "yield_7_year": 7, "yield_10_year": 10,
                "yield_20_year": 20, "yield_30_year": 30,
            }
            for k, years in mapping.items():
                y = item.get(k)
                if y is not None:
                    rows.append({"date": d, "maturity_years": years, "yield": float(y)})
        next_url = data.get("next_url")
        if not next_url:
            break
        url = next_url + f"&apiKey={POLYGON_API_KEY}"

    df = pd.DataFrame(rows).sort_values(["date", "maturity_years"])
    return df

# Example: get the latest available date <= selected_date and build the curve
df_all = load_yield_curve_polygon()
selected_date = df_all["date"].max() if "asof" not in locals() else min(df_all["date"].max(), asof)
curve_today = df_all[df_all["date"] == selected_date]