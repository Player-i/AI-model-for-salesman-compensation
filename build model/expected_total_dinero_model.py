import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import joblib

# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv('data/raw_data_ruta_01.csv')

# Step 2: Clean up column names
# Remove special characters, trim spaces, and convert to lowercase with underscores
df.columns = df.columns.str.strip().str.replace('*', '', regex=False).str.replace(' ', '_').str.lower()

# Ensure column names are correct
df.columns = df.columns.str.strip().str.replace('*', '', regex=False).str.replace(' ', '_').str.lower()

# Convert 'total_dinero' to numeric, removing any commas and spaces
df['total_dinero'] = df['total_dinero'].str.replace(',', '').str.strip().astype(float)

# Step 3: Extract the month from the 'mm-yyyy' column
df['month'] = df['mm-yyyy'].apply(lambda x: int(x.split('-')[0]))

# Step 4: Create cyclical features for the month
df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

# Step 5: Define features and target variable
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

# Ensure all required columns are present
required_columns = [
    'cantidad_de_ordenes', 'ofertas_aplicadas', 'sku_activados', 
    'cantidad_facturas_venta', 'sku_x_factura', 'ros_factura', 
    'clientes_activados', 'sin_month', 'cos_month'
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise KeyError(f"Missing columns in the DataFrame: {missing_columns}")

X = df[features]
y = df['total_dinero']        # Target variable

# Step 6: Train the Ridge regression model
model = Ridge(alpha=1.0)
model.fit(X, y)

# Save the trained model to a file
joblib.dump(model, 'model/ridge_model.pkl')

# Step 7: Make predictions and add them to the DataFrame
df['expected_total_dinero'] = model.predict(X)

# Step 8: Display the results
print("### Model Coefficients ###")
print(f"Intercept: {model.intercept_:.2f}")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.2f}")

print("\n### Predictions ###")
print(df[['mm-yyyy', 'total_dinero', 'expected_total_dinero']].to_string(index=False))