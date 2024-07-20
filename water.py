

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def water_quality_prediction(input_data):
    # Load dataset
    df = pd.read_csv("water_potability.csv")
    # Fill missing values
    df.fillna(df.mean(), inplace=True)
    # Separate features and target
    x = df.drop('Potability', axis=1)
    y = df['Potability']
    
    # Standardize features
    std_scaler = StandardScaler()
    x_scaled = std_scaler.fit_transform(x)
    
    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize KNeighborsClassifier model
    knn_model = KNeighborsClassifier()
    
    # Train the model
    knn_model.fit(x_train, y_train)
    
    # Scale the input data using the pre-defined StandardScaler
    scaled_data = std_scaler.transform([input_data])
    
    # Predict water quality using the trained KNeighborsClassifier model
    model_prediction = knn_model.predict(scaled_data)
    
    if model_prediction[0] == 0:
        return "Alert: Water Quality is Unfavorable"
    else:
        return "Quality of Water is Optimal"

