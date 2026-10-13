import streamlit as st
import pandas as pd
import joblib

st.title("Starbucks Calories Predictor")

from pathlib import Path
model = saved_data["model"]
features = saved_data["features"]

df = pd.read_csv("starbucks.csv")
df.columns = df.columns.str.strip()

drink = st.selectbox("Select Beverage", sorted(df["Beverage"].dropna().unique()))
prep = st.selectbox("Select Beverage Preparation", sorted(df["Beverage_prep"].dropna().unique()))

row = df[(df["Beverage"] == drink) & (df["Beverage_prep"] == prep)]

if st.button("Predict Calories"):
    if row.empty:
        st.warning("Beverage not found.")
    else:
        input_row = row.drop(columns=["Calories"])
        input_row = pd.get_dummies(input_row, drop_first=True, dtype=int)
        input_row = input_row.reindex(columns=features, fill_value=0)
        prediction = model.predict(input_row)[0]
        st.success(f"Predicted Calories: {prediction:.0f}")
