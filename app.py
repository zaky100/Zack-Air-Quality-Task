import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import os

# Setting up the page first
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

# Data caching with safety check
@st.cache_data
def load_data():
    # UPDATED: Matches the exact filename you uploaded to GitHub
    if os.path.exists('Task1_Combined_Data (1).csv'):
        return pd.read_csv('Task1_Combined_Data (1).csv')
    else:
        return pd.DataFrame()

# Model caching
@st.cache_resource
def load_ml_components():
    
    model = joblib.load('rf_model.pkl.gz')
    scaler = joblib.load('scaler (1).pkl')
    model_columns = joblib.load('model_columns (3).pkl')
    return model, scaler, model_columns

df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Dataset Overview", "Visualizations", "Model Outputs"])

# Check if data loaded successfully for the first two pages
if df.empty and page in ["Dataset Overview", "Visualizations"]:
    
    st.error(" Data file 'Task1_Combined_Data.csv' not found. Please ensure it is uploaded to the GitHub repository.")
else:
    # DATASET OVERVIEW
    if page == "Dataset Overview":
        st.title("Dataset Overview")
        st.write("This section allows you to explore the raw and cleaned air quality data from Beijing.")

        # Station Filter
        station_choice = st.selectbox("Filter by Station:", df['station'].unique())
        filtered_df = df[df['station'] == station_choice]

        st.write(f"Showing data for **{station_choice}** (First 100 rows):")
        st.dataframe(filtered_df.head(100))

        st.write("### Statistical Summary")
        st.write(filtered_df.describe())

    # VISUALIZATIONS
    elif page == "Visualizations":
        st.title("Data Visualizations")
        st.write("Explore relationships between pollutants and meteorological factors.")

        pollutant = st.selectbox("Select Pollutant:", ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'])

        st.subheader(f"{pollutant} Distribution Across Stations")
        fig, ax = plt.subplots(figsize=(10, 4))
        # Using the theme's background to make plots look seamless
        fig.patch.set_facecolor('#FDF6E3')
        ax.set_facecolor('#FFFFFF')
        sns.boxplot(x='station', y=pollutant, data=df, ax=ax, palette="mako")
        st.pyplot(fig)

        st.subheader(f"Temperature vs {pollutant}")
        # Safe sampling to prevent crashes if dataset is small
        sample_df = df.sample(min(2000, len(df)), random_state=42)
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        fig2.patch.set_facecolor('#FDF6E3')
        ax2.set_facecolor('#FFFFFF')
        sns.scatterplot(x='TEMP', y=pollutant, hue='category', data=sample_df, alpha=0.6, ax=ax2, palette="husl")
        st.pyplot(fig2)

# MODEL OUTPUTS (Runs independently of the CSV data)
if page == "Model Outputs":
    st.title("Predictive Model (PM2.5)")
    st.write("Adjust the weather and time conditions below to predict the PM2.5 pollution level.")

    try:
        # Load the cached model
        model, scaler, model_columns = load_ml_components()

        # Wrapped in a Form so the app doesn't lag when sliders are moved
        with st.form("prediction_form"):
            # Creating sliders for user input
            col1, col2, col3 = st.columns(3)
            with col1:
                temp = st.slider("Temperature (°C)", -20.0, 40.0, 10.0)
                pres = st.slider("Pressure (hPa)", 990.0, 1040.0, 1015.0)
                dewp = st.slider("Dew Point (°C)", -30.0, 30.0, 0.0)
            with col2:
                rain = st.slider("Rainfall (mm)", 0.0, 50.0, 0.0)
                wspm = st.slider("Wind Speed (m/s)", 0.0, 10.0, 2.0)
                month = st.slider("Month", 1, 12, 6)
            with col3:
                hour = st.slider("Hour of Day", 0, 23, 12)
                category = st.selectbox("Station Category", ['Inner (Urban)', 'Outer (Suburban)'])
                wd = st.selectbox("Wind Direction", ['N', 'E', 'S', 'W', 'NW', 'NE', 'SW', 'SE'])

            submit_button = st.form_submit_button("Predict PM2.5 Level")

        # Action taken only after clicking the button
        if submit_button:
            # Add static average values for other pollutants just to make a prediction
            pm10, so2, no2, co, o3 = 80.0, 15.0, 40.0, 1000.0, 50.0

            # Create a dataframe for the user input
            input_dict = {
                'PM10': pm10, 'SO2': so2, 'NO2': no2, 'CO': co, 'O3': o3,
                'TEMP': temp, 'PRES': pres, 'DEWP': dewp, 'RAIN': rain, 'WSPM': wspm,
                'Month': month, 'Hour': hour, 'wd': wd, 'category': category
            }
            input_df = pd.DataFrame([input_dict])

            # One-Hot Encode
            input_encoded = pd.get_dummies(input_df)
            input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

            input_scaled = scaler.transform(input_encoded)
            prediction = model.predict(input_scaled)

            st.success(f"### Predicted PM2.5 Level: {prediction[0]:.2f} μg/m³")

            if prediction[0] < 35:
                st.info(" Air Quality is Good")
            elif prediction[0] < 75:
                st.warning(" Air Quality is Moderate")
            else:
                st.error(" Air Quality is Unhealthy")

    except Exception as e:
        st.error(f"Error loading model: {e}. Make sure you ran Task 3 to generate the .pkl files and uploaded them to GitHub.")
