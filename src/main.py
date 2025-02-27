import os
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import io

from ml_utility import (read_data, read_custom_data, preprocess_data, train_model, evaluate_model, plot_confusion_matrix, plot_feature_importance)


# Get the working directory of the main.py file
working_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(working_dir)


# Set page configuration
st.set_page_config(
    page_title="TrainWise",
    page_icon="🧠",
    layout="wide"
)


# Custom CSS for modern and professional look
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
    :root {
        --bg-color-light: white;
        --text-color-light: black;
        --bg-color-dark: #0e1117;
        --text-color-dark: white;
    }
    body[data-theme="light"] {
        background-color: var(--bg-color-light);
        color: var(--text-color-light);
    }
    body[data-theme="dark"] {
        background-color: var(--bg-color-dark);
        color: var(--text-color-dark);
    }
    body {
        transition: background 0.3s, color 0.3s;
    }
    .stButton button {
        background-color: #2E86C1;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #1C6EA4;
    }
    .stHeader {
        font-size: 2.5em;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 20px;
    }
    .stSubheader {
        font-size: 1.8em;
        font-weight: bold;
        color: #2E86C1;
        margin-bottom: 15px;
    }
    .stInstruction {
        font-size: 20px;
        color: inherit;
        line-height: 1.6;
        margin-bottom: 20px;
    }
    .stMetric {
        font-size: 1.2em;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
    }
    .stMetricContainer {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 20px;
    }
    .stTable {
        width: 100%;
        font-size: 16px;
    }
    .stTable th {
        background-color: #2E86C1;
        color: white;
        font-weight: bold;
        padding: 10px;
        text-align: left;
    }
    .stTable td {
        padding: 10px;
        border-bottom: 1px solid #ddd;
    }
    button {
        padding: 10px 20px;
        cursor: pointer;
    }
    </style>
    """, 
    unsafe_allow_html=True
)


# Navigation bar
with st.sidebar:
    st.markdown("<div class='stSubheader'><i class='fas fa-rocket'></i> Navigation</div>", unsafe_allow_html=True)
    selected_page = option_menu(
        menu_title=None,
        options=["Home", "Data & Model", "Data Exploration", "Results"],
        icons=["house", "database", "search", "bar-chart"],
        default_index=0,
    )


# Home Page
if selected_page == "Home":
    st.markdown("<div class='stHeader'>🌟TrainWise: Zero Code, Maximum Impact!</div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='stInstruction'>
        Welcome to <strong>TrainWise</strong>, your no-code machine learning model training app designed specifically for <strong>classification tasks</strong>! 
        With TrainWise, you can effortlessly build and evaluate machine learning models without writing a single line of code. Here's how it works:
        </div>
        <div class='stInstruction'>
        1. <strong>Upload your dataset</strong> or choose from our default datasets.<br>
        2. <strong>Select a target column</strong> and configure your model.<br>
        3. <strong>Train the model</strong> and analyze its performance.<br>
        </div>
        <div class='stInstruction'>
        Whether you're a beginner or an expert, TrainWise simplifies the process of building classification models, making machine learning accessible to everyone.
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div class='stInstruction'>Next Step: Go to the <strong>Data & Model</strong> page to upload or select a dataset and train your model.</div>", unsafe_allow_html=True)


# Data & Model Page
if selected_page == "Data & Model":
    st.markdown("<div class='stSubheader'><i class='fas fa-database'></i> Data & Model</div>", unsafe_allow_html=True)

    # Dataset Section
    st.markdown("<div class='stInstruction'>Upload or Select Dataset</div>", unsafe_allow_html=True)
    data_source = st.radio(
        "Choose data source:",
        ("Select from default datasets", "Upload your own file")
    )

    df = None
    if data_source == "Select from default datasets":
        dataset_list = os.listdir(f"{parent_dir}/data")
        dataset = st.selectbox("Select a dataset from the dropdown", dataset_list, index=None)
        if dataset:
            df = read_data(dataset)
            st.session_state.dataset_name = dataset
    else:
        uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded_file:
            df = read_custom_data(uploaded_file)
            st.session_state.dataset_name = uploaded_file.name

    if df is not None:
        st.session_state.df = df
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head())

        # Model Training Section
        st.markdown("<div class='stInstruction'>Model Training</div>", unsafe_allow_html=True)
        target_column = st.selectbox("Choose the target column", list(df.columns))

        col1, col2 = st.columns(2)
        with col1:
            scaler_type_list = ["standard", "minmax"]
            scaler_type = st.selectbox("Select a scaler", scaler_type_list)
        with col2:
            model_dictionary = {
                "Logistic Regression": LogisticRegression(),
                "Support Vector Classifier": SVC(),
                "Random Forest Classifier": RandomForestClassifier(),
                "XGBoost Classifier": XGBClassifier()
            }
            selected_model = st.selectbox("Select a Model", list(model_dictionary.keys()))

        model_name = st.text_input("Model name", placeholder="Enter a name for your model (e.g., model.pkl)")

        if st.button("Train the Model"):
            with st.spinner("Training the model..."):
                X_train, X_test, y_train, y_test = preprocess_data(df, target_column, scaler_type)
                model_to_be_trained = model_dictionary[selected_model]
                model = train_model(X_train, y_train, model_to_be_trained, model_name)
                accuracy, precision, recall, f1, suggestions, conf_matrix = evaluate_model(model, X_test, y_test)

                st.session_state.accuracy = accuracy
                st.session_state.precision = precision
                st.session_state.recall = recall
                st.session_state.f1 = f1
                st.session_state.suggestions = suggestions
                st.session_state.conf_matrix = conf_matrix
                st.session_state.model = model
                st.session_state.feature_names = X_train.columns.tolist()

            st.success("Model training completed!")
            st.balloons()
            st.markdown("---")
            st.markdown("<div class='stInstruction'>Next Step: Go to the <strong>Results</strong> page to view the model's performance and download the trained model.</div>", unsafe_allow_html=True)


# Data Exploration Page
elif selected_page == "Data Exploration":
    st.markdown("<div class='stSubheader'><i class='fas fa-search'></i> Data Exploration</div>", unsafe_allow_html=True)
    if 'df' not in st.session_state:
        st.warning("Please upload or select a dataset on the **Data & Model** page.")
    else:
        df = st.session_state.df

        # Display dataset information
        st.subheader("📋 Dataset Information")
        st.markdown("<div class='stMetricContainer'>", unsafe_allow_html=True)
        st.markdown(f"<div class='stMetric'>Number of Rows: {df.shape[0]}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stMetric'>Number of Columns: {df.shape[1]}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stMetric'>Missing Values: {df.isnull().sum().sum()}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Attribute Information Table
        st.subheader("📋 Attribute Information")
        attribute_info = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": df.dtypes,
            "Missing Values": df.isnull().sum(),
            "Unique Values": df.nunique()
        })
        st.markdown("<div class='stTable'>", unsafe_allow_html=True)
        st.table(attribute_info)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        st.subheader("📊 Data Visualizations")
        
        # Dropdown to select the type of graph
        graph_type = st.selectbox(
            "Select the type of graph",
            ["Histogram", "Scatter Plot", "Box Plot", "Pair Plot"]
        )

        if graph_type == "Histogram":
            st.write("**Histogram**")
            column = st.selectbox("Select a column for histogram", df.columns)
            fig, ax = plt.subplots()
            sns.histplot(df[column], kde=True, ax=ax)
            st.pyplot(fig)

        elif graph_type == "Scatter Plot":
            st.write("**Scatter Plot**")
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Select X-axis", df.columns)
            with col2:
                y_axis = st.selectbox("Select Y-axis", df.columns)
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
            st.pyplot(fig)

        elif graph_type == "Box Plot":
            st.write("**Box Plot**")
            column = st.selectbox("Select a column for box plot", df.columns)
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=column, ax=ax)
            st.pyplot(fig)

        elif graph_type == "Pair Plot":
            st.write("**Pair Plot**")
            st.warning("Pair plots can take a while to render for large datasets.")
            columns = st.multiselect("Select columns for pair plot", df.columns)
            if columns:
                fig = sns.pairplot(df[columns])
                st.pyplot(fig)
            else:
                st.warning("Please select at least two columns for the pair plot.")

        st.markdown("---")
        st.markdown("<div class='stInstruction'>Next Step: Go to the <strong>Results</strong> page to view the model's performance.</div>", unsafe_allow_html=True)


# Results Page
elif selected_page == "Results":
    st.markdown("<div class='stSubheader'><i class='fas fa-chart-bar'></i> Results</div>", unsafe_allow_html=True)
    if 'accuracy' not in st.session_state:
        st.warning("Please train a model on the **Data & Model** page.")
    else:
        # Show dataset name
        if 'dataset_name' in st.session_state:
            st.markdown(f"<div class='stInstruction'><strong>Dataset:</strong> {st.session_state.dataset_name}</div>", unsafe_allow_html=True)

        st.subheader("📊 Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{st.session_state.accuracy:.2f}")
        with col2:
            st.metric("Precision", f"{st.session_state.precision:.2f}")
        with col3:
            st.metric("Recall", f"{st.session_state.recall:.2f}")
        with col4:
            st.metric("F1 Score", f"{st.session_state.f1:.2f}")

        st.subheader("📊 Confusion Matrix")
        conf_matrix_plot = plot_confusion_matrix(st.session_state.conf_matrix)
        st.pyplot(conf_matrix_plot)

        st.subheader("📊 Feature Importance")
        feature_importance_plot = plot_feature_importance(st.session_state.model, st.session_state.feature_names)
        if feature_importance_plot:
            st.pyplot(feature_importance_plot)
        else:
            st.warning("Feature importance is not available for this model.")

        st.subheader("💡 Suggestions for Improvement")
        for suggestion in st.session_state.suggestions:
            st.info(suggestion)

        st.subheader("📥 Download Trained Model")
        model_name = st.text_input("Enter a name for the model file (e.g., model.pkl)", placeholder="model.pkl")
        if st.button("Download Model"):
            # Save the model to a BytesIO object
            model_bytes = io.BytesIO()
            joblib.dump(st.session_state.model, model_bytes)
            model_bytes.seek(0)

            # Provide the download button
            st.download_button(
                label="Download Model",
                data=model_bytes,
                file_name=model_name,
                mime="application/octet-stream"
            )
            st.success(f"Model saved as {model_name}!")
        st.markdown("---")
        st.markdown("<div class='stInstruction'>Next Step: Go to the <strong>Data Exploration</strong> page to explore the dataset further.</div>", unsafe_allow_html=True)