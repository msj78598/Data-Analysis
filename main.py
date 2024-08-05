import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

# Function to load data
@st.cache_data
def load_data(file_path):
    data = pd.read_excel(file_path)
    return data

# Function to add anomaly detection columns based on the provided rules
def add_anomaly_columns(data):
    data['Voltage_Zero_Anomaly'] = (
        ((data['v1'] == 0) | (data['v2'] == 0) | (data['v3'] == 0)) & 
        ((data['a1'] != 0) | (data['a2'] != 0) | (data['a3'] != 0))
    )
    data['Current_Zero_Anomaly'] = (
        ((data['v1'] == 0) & (data['a1'] == 0) & ((data['a2'] > 0) | (data['a3'] > 0))) |
        ((data['v2'] == 0) & (data['a2'] == 0) & ((data['a1'] > 0) | (data['a3'] > 0))) |
        ((data['v3'] == 0) & (data['a3'] == 0) & ((data['a1'] > 0) | (data['a2'] > 0)))
    )
    return data

# Function to add feature engineering columns
def add_features(data):
    data['mean_v'] = data[['v1', 'v2', 'v3']].mean(axis=1)
    data['mean_a'] = data[['a1', 'a2', 'a3']].mean(axis=1)
    data['std_v'] = data[['v1', 'v2', 'v3']].std(axis=1)
    data['std_a'] = data[['a1', 'a2', 'a3']].std(axis=1)
    data['v1_v2_ratio'] = data['v1'] / (data['v2'] + 1e-6)  # Adding small value to avoid division by zero
    data['v1_v3_ratio'] = data['v1'] / (data['v3'] + 1e-6)
    data['v2_v3_ratio'] = data['v2'] / (data['v3'] + 1e-6)
    data['a1_a2_ratio'] = data['a1'] / (data['a2'] + 1e-6)
    data['a1_a3_ratio'] = data['a1'] / (data['a3'] + 1e-6)
    data['a2_a3_ratio'] = data['a2'] / (data['a3'] + 1e-6)
    return data

# Function to prepare data for classification
def prepare_classification_data(data):
    data['Anomaly'] = data['Voltage_Zero_Anomaly'] | data['Current_Zero_Anomaly']
    features = ['v1', 'v2', 'v3', 'a1', 'a2', 'a3', 'mean_v', 'mean_a', 'std_v', 'std_a', 
                'v1_v2_ratio', 'v1_v3_ratio', 'v2_v3_ratio', 'a1_a2_ratio', 'a1_a3_ratio', 'a2_a3_ratio']
    X = data[features]
    y = data['Anomaly']
    return X, y

# Function to calculate regression metrics and return the model
def calculate_regression_metrics(data, v_col, a_col):
    valid_mask = (data[v_col] != 0) & (data[a_col] != 0)
    X = data[valid_mask][[v_col]].values
    y = data[valid_mask][a_col].values
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    return mae, rmse, r2, model

# Streamlit app
st.title('Anomaly Detection in Electrical Meters')

# File upload
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file is not None:
    data = load_data(uploaded_file)
    data = add_anomaly_columns(data)
    data = add_features(data)
    X, y = prepare_classification_data(data)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Random Forest Classifier
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    # Calculate accuracy and classification report
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
    
    st.subheader('Classification Results')
    st.write(f'Accuracy of Random Forest: {accuracy_rf:.4f}')
    st.write(report_rf)
    
    # Save the anomalies to a new Excel file
    anomalies = data[data['Anomaly'] == True]
    anomalies_file_path = 'anomalies.xlsx'
    anomalies.to_excel(anomalies_file_path, index=False)
    st.subheader('Anomalies Detected')
    st.write(anomalies)
    st.markdown(f"[Download Anomalies Excel File](anomalies.xlsx)")

    # Plot regression results
    st.subheader('Regression Results')
    for v_col, a_col in [('v1', 'a1'), ('v2', 'a2'), ('v3', 'a3')]:
        mae, rmse, r2, model = calculate_regression_metrics(data, v_col, a_col)
        st.write(f'{v_col} vs {a_col} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')
        
        fig, ax = plt.subplots()
        valid_mask = (data[v_col] != 0) & (data[a_col] != 0)
        X = data[valid_mask][[v_col]].values
        y = data[valid_mask][a_col].values
        y_pred = model.predict(X)
        
        ax.scatter(X, y, label='Actual Data', color='blue')
        sorted_idx = X.flatten().argsort()
        ax.plot(X[sorted_idx], y_pred[sorted_idx], color='green', label='Regression Line')
        ax.set_xlabel(v_col)
        ax.set_ylabel(a_col)
        ax.set_title(f'Relationship between {v_col} and {a_col}')
        ax.legend()
        
        st.pyplot(fig)
