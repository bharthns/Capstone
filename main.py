import streamlit as st
import numpy as np
import pandas as pd
from model import NO2DownscalingModel
from utils import (
    load_satellite_data,
    load_ground_data,
    create_no2_map,
    calculate_metrics,
    handle_missing_data,
    save_satellite_data,
    save_ground_measurements,
    format_downscaled_csv
)
from database import init_db, get_db

# Initialize database
init_db()

st.set_page_config(
    page_title="NO2 Map Downscaling",
    page_icon="🌍",
    layout="wide"
)

# Load custom CSS
try:
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass  # CSS file is optional

st.title("🌍 NO2 Satellite Data Downscaling")
st.markdown("""
This application uses machine learning to enhance the resolution of satellite-based NO2 measurements.
Upload your data to get started!
""")

scale_factor = st.sidebar.slider(
    "Upscaling factor",
    min_value=1,
    max_value=4,
    value=2,
    help="How many times to increase the resolution in each dimension"
)
value_scale = 1e15

# File upload section
col1, col2 = st.columns(2)
with col1:
    satellite_file = st.file_uploader(
        "Upload Satellite Data (GeoTIFF)",
        type=['tif', 'tiff'],
        help="Upload satellite NO2 data in GeoTIFF format"
    )

with col2:
    ground_file = st.file_uploader(
        "Upload Ground Station Data (CSV)",
        type=['csv'],
        help="Upload ground station measurements for validation"
    )

if satellite_file is not None:
    # Load and process satellite data
    data, transform, crs = load_satellite_data(satellite_file)

    if data is not None:
        st.subheader("Input Data Visualization")

        # Handle missing data
        processed_data = handle_missing_data(data)

        # Display original map
        st.plotly_chart(
            create_no2_map(processed_data, "Original NO2 Concentration", value_scale=value_scale),
            use_container_width=True
        )

        # Save data to database
        db = next(get_db())
        save_satellite_data(db, processed_data, transform)

        if ground_file is not None:
            ground_data = load_ground_data(ground_file)
            if ground_data is not None:
                save_ground_measurements(db, ground_data)

        # Model training and prediction
        with st.spinner("Training model and generating high-resolution map..."):
            model = NO2DownscalingModel()
            X_val, y_val = model.train(processed_data)

            # Generate high-resolution map
            downscaled_data = model.predict(processed_data, scale_factor=scale_factor)

            st.subheader("Downscaled Map")
            st.plotly_chart(
                create_no2_map(downscaled_data, "Downscaled NO2 Concentration", value_scale=value_scale),
                use_container_width=True
            )

            # Calculate and display metrics
            metrics = calculate_metrics(y_val, model.model.predict(X_val))

            st.subheader("Model Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Squared Error", f"{metrics['MSE']:.4f}")
            col2.metric("Root Mean Squared Error", f"{metrics['RMSE']:.4f}")
            col3.metric("R² Score", f"{metrics['R2']:.4f}")

            # Download results
            st.download_button(
                label="Download Downscaled Map",
                data=format_downscaled_csv(
                    data=downscaled_data,
                    transform=transform,
                    scale_factor=scale_factor,
                    value_scale=value_scale
                ),
                file_name="downscaled_no2_map.csv",
                mime="text/csv"
            )
            st.caption("CSV columns: row, col, latitude, longitude, raw NO₂, NO₂ scaled (×10¹⁵ mol/cm²).")
else:
    st.info("Please upload satellite data to begin the downscaling process.")

# Add footer
st.markdown("---")
st.markdown(
    "Data source: Satellite NO2 measurements"
)