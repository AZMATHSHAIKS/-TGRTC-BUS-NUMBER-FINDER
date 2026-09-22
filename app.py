import streamlit as st
import pandas as pd
import joblib
import pickle
model = joblib.load("route_model.pkl")
encoders = pickle.load(open("encoders.pkl", "rb"))
print(encoders.keys())
st.set_page_config(page_title="TGRTC BUS NUMBER FINDER")

st.title("🚌 TGRTC BUS NUMBER FINDER")
st.write("Enter your starting point and destination to find your bus number.")

from_val = st.text_input("From (origin stop)")
to_val = st.text_input("To (destination stop)")

if st.button("Find Bus Number"):
    try:
        f_enc = encoders['from'].transform([from_val.upper()])[0]
        t_enc = encoders['destination'].transform([to_val.upper()])[0]
        pred = model.predict([[f_enc, t_enc]])
        route = encoders['bus_number'].inverse_transform(pred)[0]
        st.success(f"Bus Number: {route}")
    except ValueError:
        st.error("One of these stops wasn't seen during training. Try a different spelling.")