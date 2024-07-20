import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv("air_quality.csv")

# Convert continuous target variable into categories
df['AQI_category'] = pd.cut(df['AQI'], bins=[-np.inf, 50, np.inf], labels=['Good', 'Polluted'])

# Separate features and target variable
x = df.drop(['AQI', 'AQI_category'], axis=1)
y = df['AQI_category']

# Standardize features
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Instantiate SVM classifier
model_lr = SVC(kernel='rbf', random_state=42)

# Train the model
model_lr.fit(x_train, y_train)

# Make predictions on the test set
pred_lr = model_lr.predict(x_test)

# Calculate accuracy score
accuracy_score_lr = accuracy_score(y_test, pred_lr)

try:
    with open('air_model.pkl', 'wb') as f:
        pickle.dump(pred_lr, f)
    print("Model saved successfully!")
except Exception as e:
    print("Error saving the model:", e)
