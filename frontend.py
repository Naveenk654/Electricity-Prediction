import streamlit as st
import requests

API_URL = "https://electricity-prediction-1-rmkt.onrender.com/predict"

st.set_page_config(page_title='Electricity Cost Prediction')
st.title('Electricity Cost Prediction')
st.write("This is a simple web application that uses a machine learning model to predict the electricity cost.")
st.markdown("Enter the following parameters to calculate cost:")
with st.form("my_form"):
    site_area =st.number_input("site area")
    structure_type=st.selectbox("structure type", ['Mixed-use', 'Residential', 'Commercial', 'Industrial'])
    water_consumption  =st.number_input("water consumption")
    recycling_rate= st.number_input("recycling rate")
    utilisation_rate= st.number_input("utilisation rate")
    air_quality_index=st.number_input("air quality index")
    issue_reolution_time=st.number_input("issue reloution time")
    resident_count=st.number_input("resident_count")
    submit = st.form_submit_button("Calculate Cost")

if submit:
    if not all([site_area,structure_type,water_consumption,recycling_rate,utilisation_rate,air_quality_index,issue_reolution_time,resident_count]): 
        st.error("Please fill in all the fields")
    else:
        with st.spinner("Calculating..."):
            
            response = requests.post(API_URL, json={"site area":site_area,"structure type":structure_type,"water consumption":water_consumption,"recycling rate":recycling_rate,"utilisation rate":utilisation_rate,"air quality index":air_quality_index,"resident count":resident_count,"issue reolution time":issue_reolution_time})
            if response.status_code == 200:
              result = response.json()
              st.success(f"✅ Predicted Cost as :$ **{result['predicted_value']}**")
            else:
             st.error(f"❌ Failed to classify (Status code: {response.status_code})")
             st.json(response.json())

        