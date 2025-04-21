import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the data from the CSV file
df = pd.read_csv('data/raw_data_ruta_01.csv')

# Clean column names: remove special characters, trim spaces, and convert to lowercase with underscores
df.columns = df.columns.str.strip().str.replace('*', '', regex=False).str.replace(' ', '_').str.lower()

# Convert 'total_dinero' to numeric, handling commas and spaces
df['total_dinero'] = df['total_dinero'].str.replace(',', '').str.strip().astype(float)


# Extract month from 'mm-yyyy' for cyclical features
df['month'] = df['mm-yyyy'].apply(lambda x: int(x.split('-')[0]))

# Create cyclical features for seasonality
df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

# Define features for the model
features = [
    'cantidad_de_ordenes',    # Number of orders
    'ofertas_aplicadas',      # Applied offers
    'sku_activados',          # Activated SKUs
    'cantidad_facturas_venta',# Number of sales invoices
    'sku_x_factura',          # SKUs per invoice
    'ros_factura',            # Rate of sale per invoice
    'clientes_activados',     # Activated clients
    'sin_month',              # Cyclical sine feature
    'cos_month'               # Cyclical cosine feature
]

# Check for missing columns
missing_columns = [col for col in features if col not in df.columns]
if missing_columns:
    raise KeyError(f"Missing columns in the DataFrame: {missing_columns}")

# Prepare data for modeling
X = df[features]
y = df['total_dinero']

# Train the Random Forest regression model
model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=3)
model.fit(X, y)

# Save the model for future use
joblib.dump(model, 'model/random_forest_model.pkl')

# Predict expected total revenue
df['expected_total_dinero'] = model.predict(X)

# Display feature importances
print("### Feature Importances ###")
for feature, importance in zip(features, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

print("\n### Predictions ###")
print(df[['mm-yyyy', 'total_dinero', 'expected_total_dinero']].to_string(index=False))

# Save the DataFrame with expected_total_dinero for use in the bonus calculation
df.to_csv('data/simulated/ruta_01_with_expected_random_forest.csv', index=False)