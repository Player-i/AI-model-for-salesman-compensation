import pandas as pd
import numpy as np
import joblib

class CustomLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.feature_means = None
        self.feature_stds = None
        self.target_mean = None
        self.target_std = None
        
    def _normalize_features(self, X):
        if self.feature_means is None or self.feature_stds is None:
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0)
            # Avoid division by zero
            self.feature_stds[self.feature_stds == 0] = 1
        return (X - self.feature_means) / self.feature_stds
    
    def _normalize_target(self, y):
        if self.target_mean is None or self.target_std is None:
            self.target_mean = np.mean(y)
            self.target_std = np.std(y)
        return (y - self.target_mean) / self.target_std
    
    def _denormalize_predictions(self, y_pred):
        return y_pred * self.target_std + self.target_mean
    
    def fit(self, X, y):
        # Normalize features and target
        X_norm = self._normalize_features(X)
        y_norm = self._normalize_target(y)
        
        n_samples, n_features = X_norm.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient Descent with safeguards
        for _ in range(self.n_iterations):
            # Calculate predictions
            y_pred = np.dot(X_norm, self.weights) + self.bias
            
            # Calculate gradients with numerical stability
            error = y_pred - y_norm
            dw = (1/n_samples) * np.dot(X_norm.T, error)
            db = (1/n_samples) * np.sum(error)
            
            # Update parameters with gradient clipping
            max_grad = 1.0  # Prevent exploding gradients
            dw = np.clip(dw, -max_grad, max_grad)
            db = np.clip(db, -max_grad, max_grad)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        X_norm = self._normalize_features(X)
        y_pred_norm = np.dot(X_norm, self.weights) + self.bias
        return self._denormalize_predictions(y_pred_norm)

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

# Step 6: Train our custom linear regression model
model = CustomLinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

# Save the trained model to a file
joblib.dump(model, 'model/custom_linear_model.pkl')

# Step 7: Make predictions and add them to the DataFrame
df['expected_total_dinero'] = model.predict(X)

# Step 8: Display the results
print("### Model Coefficients ###")
print(f"Bias (Intercept): {model.bias:.2f}")
for feature, coef in zip(features, model.weights):
    print(f"{feature}: {coef:.2f}")

print("\n### Predictions ###")
print(df[['mm-yyyy', 'total_dinero', 'expected_total_dinero']].to_string(index=False))

# Save the DataFrame with expected_total_dinero for use in the bonus calculation
df.to_csv('data/ruta_01_with_expected_custom_regression.csv', index=False)