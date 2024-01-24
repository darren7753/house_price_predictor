import numpy as np
import pandas as pd
import streamlit as st
import pickle

from streamlit_gsheets import GSheetsConnection
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="House Price Predictor",
    layout="wide"
)

reduce_header_height_style = """
    <style>
        div.block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
"""
st.markdown(reduce_header_height_style, unsafe_allow_html=True)

hide_decoration_bar_style = """
    <style>
        footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

conn = st.connection("gsheets", type=GSheetsConnection)
df = conn.read(worksheet="House Data", ttl=0, usecols=list(range(6)))
estimated_cols = ["kt", "km", "grs", "lt", "lb", "estimated_price"]
if not all(col in df.columns for col in estimated_cols):
    df = pd.DataFrame(columns=estimated_cols)
else:
    df = df.dropna()

with open("Models/LinearRegression.pkl", "rb") as file:
    lr_model = pickle.load(file)

st.markdown("<h1 style='text-align: center; margin-bottom: 20px;'>House Price Predictor</h1>", unsafe_allow_html=True)

page_option = option_menu(
    menu_title=None,
    options=["Input", "House Data"],
    orientation="horizontal"
)

st.divider()

if page_option == "Input":
    col1, col2, col3, col4, col5 = st.columns(5)
    kt = col1.number_input("Bedrooms", min_value=0, step=1)
    km = col2.number_input("Bathrooms", min_value=0, step=1)
    grs = col3.number_input("Garages", min_value=0, step=1)
    lt = col4.number_input("Land Area", min_value=0.0, format="%.2f")
    lb = col5.number_input("Building Area", min_value=0.0, format="%.2f")

    if st.button("Predict"):
        if kt == 0 and km == 0 and grs == 0 and lt == 0.0 and lb == 0.0:
            st.warning("Please provide input values before predicting.")
        elif lt == 0.0 or lb == 0.0:
            st.warning("Both 'Land Area' and 'Building Area' must be greater than 0.")
        else:
            features = np.array([[kt, km, grs, lt, lb]])
            predicted_price = lr_model.predict(features)
            st.success(f"Estimated House Price: {predicted_price[0]:,.2f}")

            results = pd.DataFrame({
                "kt": [kt],
                "km": [km],
                "grs": [grs],
                "lt": [lt],
                "lb": [lb],
                "estimated_price": [predicted_price[0]]
            })
            results = pd.concat([df, results])
            conn.update(worksheet="House Data", data=results)

else:
    df_updated = conn.read(worksheet="House Data", ttl=0, usecols=list(range(6)))
    if not all(col in df_updated.columns for col in estimated_cols):
        st.warning("There hasn't been any data available yet.")
    else:
        df_updated = df_updated.dropna()
        st.dataframe(df_updated, use_container_width=True)