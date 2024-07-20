import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE 
import numpy as np
import pickle

def predict_wine_quality(input_data):
    # Import Libraries
    import warnings
    warnings.filterwarnings('ignore')

    # Read data
    df = pd.read_csv('winequality.csv')

    # Handle missing values
    for col, value in df.items():
        if col != 'type':
            df[col] = df[col].fillna(df[col].mean())

    # Prepare data
    df1 = df.drop(['type'], axis=1)
    x = df.drop(columns=['type', 'quality'])
    y = df['quality']

    # Oversampling using SMOTE
    oversample = SMOTE(k_neighbors=4)
    x, y = oversample.fit_resample(x, y)

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Sample input data for prediction
    # input_data = (5.7, 1.13, 0.09, 1.5, 0.172, 7.0, 19.0, 0.994, 3.5, 0.48, 9.8)
    input_data_as_np_array = np.asarray(input_data)
    input_data_reshape = input_data_as_np_array.reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data_reshape)

    return prediction[0]


feature_names = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "ph", "sulphates", "alcohol"]
