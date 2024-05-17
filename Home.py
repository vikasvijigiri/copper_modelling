import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

def run_notebook(notebook_path):
    try:
        # Execute the notebook
        subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace', notebook_path], check=True)
        print(f"Successfully executed the notebook: {notebook_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing the notebook: {e}")



# Load the pre-trained models from pickle files
def load_model(file_path, model_type):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"{model_type} model file not found: {file_path}")
        st.warning(f"Running the Jupyter Notebook to Generate these. Please wait few mins....")
        notebook_path = 'Copper_modelling.ipynb'  # Replace with your notebook file path
        run_notebook(notebook_path)        

        # Retry loading the model after notebook execution
        try:
            with open(file_path, 'rb') as file:
                model = pickle.load(file)
            return model
        except FileNotFoundError:
            st.error(f"{model_type} model file still not found after running the notebook: {file_path}")
            st.stop()

classifier_model = load_model('status_pred_model.pkl', 'Classifier')
regressor_model = load_model('selling_price_prediction.pkl', 'Regressor')


# @st.cache
# def load_data():
#     return pd.read_csv('Copper_Set.xlsx - Result 1.csv')


# Functions to make predictions
def predict_classifier(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = classifier_model.predict(input_data)
    return prediction[0]

def predict_regressor(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = regressor_model.predict(input_data)
    return prediction[0]

# Streamlit app
def main():
    st.set_page_config(page_title="Business Forecaster App", page_icon=":crystal_ball:", layout="centered")

    st.title("Prediction App")
    st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.header("Model Selection")
    model_type = st.sidebar.selectbox("Choose Model Type", ["Classifier", "Regressor"])

    if model_type == "Classifier":
        st.header("Classifier Input Parameters")

        # Upload file section
        uploaded_file = st.file_uploader("Upload a CSV file with input parameters", type=["csv"], key="classifier")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data")
            st.write(data)

            if st.button("Predict from file"):
                predictions = data.apply(lambda row: predict_classifier(row.values), axis=1)
                data['Prediction'] = predictions
                st.write("Predictions")
                st.write(data)

                # Plotting the predictions
                st.header("Plot Options")
                plot_type = st.selectbox("Select plot type", ["Bar Plot", "Line Plot", "Scatter Plot"], key="classifier_plot")

                plt.figure(figsize=(10, 6))
                if plot_type == "Bar Plot":
                    sns.barplot(x=data.index, y=data['Prediction'])
                    plt.title("Bar Plot of Predictions")
                elif plot_type == "Line Plot":
                    sns.lineplot(x=data.index, y=data['Prediction'])
                    plt.title("Line Plot of Predictions")
                elif plot_type == "Scatter Plot":
                    sns.scatterplot(x=data.index, y=data['Prediction'])
                    plt.title("Scatter Plot of Predictions")

                st.pyplot(plt)

        else:
            # Manual input section with more parameters
            col1, col2, col3, col4 = st.columns(4)
            param1 = col1.number_input("Customer ID", min_value=0, max_value=1000000000, value=30156308, key="classifier_param1")
            param2 = col2.number_input("Country", min_value=0, max_value=200, value=25, key="classifier_param2")
            param4 = col3.number_input("Item Type", min_value=0, max_value=10, value=1, key="classifier_param4")
            param5 = col4.number_input("Application", min_value=0, max_value=100, value=41, key="classifier_param5")
            param6 = col1.number_input("Width", min_value=0.0, max_value=1e5, value=1210.0, key="classifier_param6")
            param7 = col2.number_input("Product Ref", min_value=0, max_value=100000000000, value=1670798778, key="classifier_param7")
            param8 = col3.number_input("Quantity Tons", min_value=0.0, max_value=1e6, value=np.exp(6.65), key="classifier_param8")
            param9 = col4.number_input("Thickness", min_value=0.0, max_value=1e6, value=np.exp(0.65), key="classifier_param9")
            param10 = col1.number_input("Selling Price", min_value=0.0, max_value=1e6, value=np.exp(6.89), key="classifier_param10")
            param11 = col2.number_input("Item Date Day", min_value=1, max_value=31, value=1, key="classifier_param11")
            param12 = col3.number_input("Item Date Month", min_value=1, max_value=12, value=4, key="classifier_param12")
            param13 = col4.number_input("Item Date Year", min_value=2019, max_value=2030, value=2020, key="classifier_param13")
            param14 = col1.number_input("Delivery Date Day", min_value=1, max_value=31, value=1, key="classifier_param14")
            param15 = col2.number_input("Delivery Date Month", min_value=1, max_value=12, value=9, key="classifier_param15")
            param16 = col3.number_input("Delivery Date Year", min_value=2019, max_value=2030, value=2020, key="classifier_param16")

            input_data = [param1, param2, param4, param5, param6, param7, np.log(param8), np.log(param9), np.log(param10), param11, param12, param13, param14, param15, param16]

            if st.button("Predict", key="classifier_predict"):
                result = predict_classifier(input_data)
                st.success(f"The predicted class is: {result}")

    elif model_type == "Regressor":
        st.header("Regressor Input Parameters")

        # Upload file section
        uploaded_file = st.file_uploader("Upload a CSV file with input parameters", type=["csv"], key="regressor")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data")
            st.write(data)

            if st.button("Predict from file"):
                predictions = data.apply(lambda row: predict_regressor(row.values), axis=1)
                data['Prediction'] = predictions
                st.write("Predictions")
                st.write(data)

                # Plotting the predictions
                st.header("Plot Options")
                plot_type = st.selectbox("Select plot type", ["Bar Plot", "Line Plot", "Scatter Plot"], key="regressor_plot")

                plt.figure(figsize=(10, 6))
                if plot_type == "Bar Plot":
                    sns.barplot(x=data.index, y=data['Prediction'])
                    plt.title("Bar Plot of Predictions")
                elif plot_type == "Line Plot":
                    sns.lineplot(x=data.index, y=data['Prediction'])
                    plt.title("Line Plot of Predictions")
                elif plot_type == "Scatter Plot":
                    sns.scatterplot(x=data.index, y=data['Prediction'])
                    plt.title("Scatter Plot of Predictions")

                st.pyplot(plt)

        else:

                
            # Manual input section with more parameters
            col1, col2, col3, col4 = st.columns(4)
            param1 = col1.number_input("Customer ID", min_value=0, max_value=1000000000, value=30156308, key="regressor_param1")
            param2 = col2.number_input("Country", min_value=0, max_value=200, value=25, key="regressor_param2")
            param3 = col3.number_input("Status", min_value=0, max_value=8, value=1, key="classifier_param3")
            param4 = col4.number_input("Item Type", min_value=0, max_value=10, value=1, key="regressor_param4")
            param5 = col1.number_input("Application", min_value=0, max_value=100, value=41, key="regressor_param5")
            param6 = col2.number_input("Width", min_value=0.0, max_value=1e5, value=1210.0, key="regressor_param6")
            param7 = col3.number_input("Product Ref", min_value=0, max_value=1000000000000, value=1670798778, key="regressor_param7")
            param8 = col4.number_input("Quantity Tons", min_value=0.0, max_value=1e6, value=np.exp(6.65), key="regressor_param8")
            param9 = col1.number_input("Thickness", min_value=0.0, max_value=1e6, value=np.exp(0.65), key="regressor_param9")
            param11 = col2.number_input("Item Date Day", min_value=1, max_value=31, value=1, key="regressor_param11")
            param12 = col3.number_input("Item Date Month", min_value=1, max_value=12, value=4, key="regressor_param12")
            param13 = col4.number_input("Item Date Year", min_value=2019, max_value=2030, value=2020, key="regressor_param13")
            param14 = col1.number_input("Delivery Date Day", min_value=1, max_value=31, value=1, key="regressor_param14")
            param15 = col2.number_input("Delivery Date Month", min_value=1, max_value=12, value=9, key="regressor_param15")
            param16 = col3.number_input("Delivery Date Year", min_value=2019, max_value=2030, value=2020, key="regressor_param16")

            input_data = [param1, param2, param3, param4, param5, param6, param7, np.log(param8), np.log(param9), param11, param12, param13, param14, param15, param16]

            if st.button("Predict", key="regressor_predict"):
                result = predict_regressor(input_data)
                st.success(f"The predicted value is: {np.exp(result)}")

    st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()











 # 0   customer             150428 non-null  Int64  
 # 1   country              150428 non-null  Int32  
 # 2   status               150428 non-null  int32  
 # 3   item type            150428 non-null  float64
 # 4   application          150428 non-null  Int32  
 # 5   width                150428 non-null  float64
 # 6   product_ref          150428 non-null  int64  
 # 7   quantity_tons_log    150428 non-null  float64
 # 8   thickness_log        150428 non-null  float64
 # 9   selling_price_log    150428 non-null  float64
 # 10  item_date_day        150428 non-null  int32  
 # 11  item_date_month      150428 non-null  int32  
 # 12  item_date_year       150428 non-null  int32  
 # 13  delivery_date_day    150428 non-null  int32  
 # 14  delivery_date_month  150428 non-null  int32  
 # 15  delivery_date_year   150428 non-null  int32  