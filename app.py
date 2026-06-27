import pandas as pd
import streamlit as st
import joblib

model = joblib.load("Zameen.Com-House-Price-Model.pkl")
df = pd.read_csv("data/zameen-updated.csv")

cities = sorted(df["city"].unique())
locations = sorted(df["location"].unique())
property_types = sorted(df["property_type"].unique())
area_types = sorted(df["Area Type"].unique())
purposes = sorted(df["purpose"].unique())
provinces = sorted(df["province_name"].unique())

st.title("Zameen.Com House Price Prediction")

property_type = st.selectbox("Property Type", property_types)
city = st.selectbox("City", cities)
location = st.selectbox("Location", locations)
province = st.selectbox("Province", provinces)
purpose = st.selectbox("Purpose", purposes)
area_type = st.selectbox("Area Type", area_types)
area_size = st.number_input("Area Size", min_value=1.0)
bedrooms = st.number_input("Bedrooms", min_value=1)
baths = st.number_input("Bathrooms", min_value=1)
latitude = st.number_input("Latitude", value=31.5204)
longitude = st.number_input("Longitude", value=74.3587)


if st.button("Predict Price"):
    input_df = pd.DataFrame(
        {
            "property_type": [property_type],
            "location": [location],
            "city": [city],
            "province_name": [province],
            "latitude": [latitude],
            "longitude": [longitude],
            "baths": [baths],
            "bedrooms": [bedrooms],
            "Area Type": [area_type],
            "Area Size": [area_size],
            "purpose": [purpose],
        }
    )

    prediction = model.predict(input_df)

    st.success(f"Predicted Price : PKR {prediction[0]:,.0f}")