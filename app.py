import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
page_title="Starbucks Calories Predictor",
page_icon="☕"
)

st.title("Starbucks Calories Predictor ☕")
st.write("Predict the calories of your Starbucks drink using Machine Learning.")

# Load trained model

saved_data = joblib.load("decision_tree_model.pkl")
model = saved_data["model"]
features = saved_data["features"]

# Load dataset

df = pd.read_csv("starbucks.csv")
df.columns = df.columns.str.strip()

st.subheader("Select Your Drink")

input_data = {}

for col in df.columns:
if col == "Calories":
continue

```
if pd.api.types.is_numeric_dtype(df[col]):
    value = st.number_input(
        f"Enter {col}",
        value=float(df[col].median())
    )
    input_data[col] = value

else:
    options = df[col].dropna().astype(str).unique().tolist()

    if options:
        input_data[col] = st.selectbox(
            f"Select {col}",
            options
        )
```

if st.button("Predict Calories"):
row = pd.DataFrame([input_data])

```
row = pd.get_dummies(
    row,
    drop_first=True,
    dtype=int
)

row = row.reindex(
    columns=features,
    fill_value=0
)

prediction = model.predict(row)[0]

st.success(f"Predicted Calories: {prediction:.0f} calories")
```

st.caption("Powered by Decision Tree Machine Learning")
