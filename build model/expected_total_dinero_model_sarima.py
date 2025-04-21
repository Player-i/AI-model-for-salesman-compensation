import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib

# Load the data from the CSV file
df = pd.read_csv('data/raw_data_ruta_01.csv')

# Clean column names: remove special characters, trim spaces, and convert to lowercase with underscores
df.columns = df.columns.str.strip().str.replace('*', '', regex=False).str.replace(' ', '_').str.lower()

# Convert 'total_dinero' to numeric, handling commas and spaces
df['total_dinero'] = df['total_dinero'].str.replace(',', '').str.strip().astype(float)


# Convert 'mm-yyyy' to a datetime index for time series modeling
df['date'] = pd.to_datetime(df['mm-yyyy'], format='%m-%Y')
df.set_index('date', inplace=True)
df.sort_index(inplace=True)  # Ensure the data is ordered by date

# Prepare the target variable (total_dinero) as a time series
y = df['total_dinero']

# Train the SARIMA model
# SARIMA parameters: (p, d, q)(P, D, Q, s)
# Using (1,1,1)(1,1,1,12) for monthly data with yearly seasonality
model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit(disp=False)  # Fit the model, suppressing convergence warnings

# Save the trained model
joblib.dump(model_fit, 'model/sarima_model.pkl')

# Predict expected total revenue (in-sample prediction for existing data)
df['expected_total_dinero'] = model_fit.predict(start=df.index[0], end=df.index[-1])

# Display model summary and predictions
# print("### SARIMA Model Summary ###")
# print(model_fit.summary())

print("\n### Predictions ###")
print(df[['total_dinero', 'expected_total_dinero']].to_string())
print("NEED MORE DATA FOR ACCURATE PREDCTIONS, MAYBE IN THE FUTURE")
# Save the DataFrame with predictions
df.to_csv('data/ruta_01_with_expected_sarima.csv')