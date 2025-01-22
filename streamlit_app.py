import streamlit as st
from demo.time_series_streamlit_app import time_series_main
from demo.regression_streamlit_app import regression_main
from demo.anomaly_detection_streamlit_app import anomaly_main
from demo.clustering_streamlit_app import clustering_main
from demo.classification_streamlit_app import classification_main
from PIL import Image
import base64

def get_image_base64(image_path):
    """
    Function to convert an image to base64

    Args:
        image_path: str: The path to the image file

    Returns:
        str: The base64 encoded image
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Load images
# main_image_base64 = get_image_base64('images/beertech_full_size.png')
sidebar_image = Image.open('images/AutoML_Logo.png')

def main():
    """
    The main function to run the AutoML web app
    """
    # # Display main image
    # st.markdown(
    #     f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{main_image_base64}" width="300"/></div>', 
    #     unsafe_allow_html=True,
    # )
    # Display sidebar image
    st.sidebar.image(sidebar_image, use_column_width=True)

    # Set page title and favicon
    st.markdown("""
    <style>
    .reportview-container {
        background: #f0f0f5;
    }
    .sidebar .sidebar-content {
        background: #f0f0f5;
    }
    h1 {
        color: #ff6347;
        font-size: 48px;
    }
    .stMarkdown h2 {
        color: #4682b4;
    }
    </style>
    """, unsafe_allow_html=True)
    st.sidebar.title("AutoML Menu")

    # Display the AutoML method selection dropdown menu
    app_mode = st.sidebar.selectbox("Choose the Automated ML Method",
                                    ["AutoML Time Series", "AutoML Regression", "AutoML Anomaly Detection", "AutoML Clustering", "AutoML Classification"])
    
    # Display information about the selected AutoML method
    if app_mode == "AutoML Time Series":
        st.info("""
        - Time series analysis uses statistical techniques to analyze series of data points ordered in time. 
        - **Why Use:** Time series analysis is crucial for forecasting and trend analysis, which are vital in fields like finance, economics, and business.
        - **Perks:** It allows businesses to understand patterns in their data and make informed future predictions, leading to better strategic decision-making.
        """)
        # Call the time_series_main function from the time_series_streamlit_app.py file
        time_series_main()

    elif app_mode == "AutoML Regression":
        st.info("""
        - Regression analysis estimates the relationship between a dependent variable and one or more independent variables.
        - **Why Use:** It's used when we want to predict a continuous output variable from the input variables.
        - **Perks:** It helps businesses understand how the dependent variable changes with changes in the independent variable, enabling them to make strategic decisions.
        """)
        # Call the regression_main function from the regression_streamlit_app.py file
        regression_main()

    elif app_mode == "AutoML Anomaly Detection":
        st.info("""
        - Anomaly detection identifies outliers that differ significantly from the majority of the data.
        - **Why Use:** It's used to detect abnormal behavior or rare events such as fraud detection in credit card transactions, or intrusion detection in network traffic.
        - **Perks:** It helps businesses identify potential problems early on, preventing significant losses or damages.
        """)
        # Call the anomaly_main function from the anomaly_detection_streamlit_app.py file
        anomaly_main()

    elif app_mode == "AutoML Clustering":
        st.info("""
        - Clustering divides data points into several groups so that data points in the same group are more similar to each other than to those in other groups.
        - **Why Use:** It's used when we want to understand the structure and patterns in data when we don't have a target variable for supervision.
        - **Perks:** It helps businesses understand the segmentation in their customer base, leading to more personalized marketing and better customer service.
        """)
        # Call the clustering_main function from the clustering_streamlit_app.py file
        clustering_main()

    elif app_mode == "AutoML Classification":
        st.info("""
        - Classification identifies the category an observation belongs to based on a training dataset.
        - **Why Use:** It's used when we want to predict the category of an observation.
        - **Perks:** It helps businesses predict outcomes and make data-driven decisions, leading to improved services and customer satisfaction.
        """)
        # Call the classification_main function from the classification_streamlit_app.py file
        classification_main()

if __name__ == "__main__":
    main()