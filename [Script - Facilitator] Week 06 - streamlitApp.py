import streamlit as st
import pickle
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Timelytics", page_icon=":pencil:", layout="wide")

st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")

st.caption(
    "Timelytics is an ensemble model that utilizes three powerful machine learning algorithms - XGBoost, Random Forests, and Support Vector Machines (SVM) - to accurately forecast Order to Delivery (OTD) times. By combining the strengths of these three algorithms, Timelytics provides a robust and reliable prediction of OTD times, helping businesses to optimize their supply chain operations."
)

st.caption(
    "With Timelytics, businesses can identify potential bottlenecks and delays in their supply chain and take proactive measures to address them, reducing lead times and improving delivery times. The model utilizes historical data on order processing times, production lead times, shipping times, and other relevant variables to generate accurate forecasts of OTD times. These forecasts can be used to optimize inventory management, improve customer service, and increase overall efficiency in the supply chain."
)

# Cache the model so it is not reloaded every time a widget changes
@st.cache_resource
def load_model():
    with open("./voting_model.pkl", "rb") as f:
        return pickle.load(f)

voting_model = load_model()

# Build prediction input as a DataFrame with feature names
def waitime_predictor(
    purchase_dow,
    purchase_month,
    year,
    product_size_cm3,
    product_weight_g,
    geolocation_state_customer,
    geolocation_state_seller,
    distance,
):
    input_df = pd.DataFrame([{
        "Purchased Day of the Week": purchase_dow,
        "Purchased Month": purchase_month,
        "Purchased Year": year,
        "Product Size in cm^3": product_size_cm3,
        "Product Weight in grams": product_weight_g,
        "Geolocation State Customer": geolocation_state_customer,
        "Geolocation State Seller": geolocation_state_seller,
        "Distance": distance,
    }])

    prediction = voting_model.predict(input_df)
    return round(float(prediction[0]), 2)

with st.sidebar:
    img = Image.open("./assets/supply_chain_optimisation.jpg")
    st.image(img)
    st.header("Input Parameters")

    purchase_dow = st.number_input(
        "Purchased Day of the Week", min_value=0, max_value=6, step=1, value=3
    )
    purchase_month = st.number_input(
        "Purchased Month", min_value=1, max_value=12, step=1, value=1
    )
    year = st.number_input("Purchased Year", min_value=2016, max_value=2025, step=1, value=2018)

    # Add reasonable bounds to avoid wild inputs
    product_size_cm3 = st.number_input(
        "Product Size in cm^3", min_value=1.0, max_value=200000.0, value=9328.0
    )
    product_weight_g = st.number_input(
        "Product Weight in grams", min_value=1.0, max_value=50000.0, value=1800.0
    )
    geolocation_state_customer = st.number_input(
        "Geolocation State of the Customer", min_value=0, max_value=30, step=1, value=10
    )
    geolocation_state_seller = st.number_input(
        "Geolocation State of the Seller", min_value=0, max_value=30, step=1, value=20
    )
    distance = st.number_input(
        "Distance", min_value=0.0, max_value=5000.0, value=475.35
    )

    submit = st.button("Predict Wait Time!")

with st.container():
    st.header("Output: Wait Time in Days")

    if submit:
        try:
            with st.spinner("This may take a moment..."):
                prediction = waitime_predictor(
                    purchase_dow,
                    purchase_month,
                    year,
                    product_size_cm3,
                    product_weight_g,
                    geolocation_state_customer,
                    geolocation_state_seller,
                    distance,
                )
            st.success(f"Predicted wait time: {prediction} days")
        except Exception as e:
            st.exception(e)

    data = {
        "Purchased Day of the Week": ["0", "3", "1"],
        "Purchased Month": ["6", "3", "1"],
        "Purchased Year": ["2018", "2017", "2018"],
        "Product Size in cm^3": ["37206.0", "63714", "54816"],
        "Product Weight in grams": ["16250.0", "7249", "9600"],
        "Geolocation State Customer": ["25", "25", "25"],
        "Geolocation State Seller": ["20", "7", "20"],
        "Distance": ["247.94", "250.35", "4.915"],
    }

    df = pd.DataFrame(data)
    st.header("Sample Dataset")
    st.write(df)
