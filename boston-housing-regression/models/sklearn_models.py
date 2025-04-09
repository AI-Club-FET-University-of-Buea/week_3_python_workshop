import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_boston_housing_data(filepath):
    """Load the Boston Housing dataset from a CSV file."""
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """Preprocess the dataset by handling missing values and splitting into features and target."""
    # Check for missing values
    if df.isnull().sum().any():
        df.fillna(df.mean(), inplace=True)  # Fill missing values with mean
    
    X = df.drop('MEDV', axis=1)  # Features
    y = df['MEDV']  # Target variable
    return X, y

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def build_linear_regression_model(X_train, y_train):
    """Build and train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using Mean Squared Error and R-squared metrics."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def get_model_coefficients(model):
    """Get the coefficients of the trained Linear Regression model."""
    return model.coef_, model.intercept_