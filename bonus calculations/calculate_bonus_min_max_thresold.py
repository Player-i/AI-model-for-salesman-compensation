import pandas as pd
import numpy as np

# Load your DataFrame
df = pd.read_csv('data/ruta_01_with_expected_ridge_regression.csv') 

# Calculate the ratio
df['ratio'] = df['total_dinero'] / df['expected_total_dinero']

# Define bonus parameters
threshold_min = 0.98  # Start bonus at 90%
threshold_max = 1.035  # Full bonus at 103.5%
max_bonus = 50000

# Calculate the bonus
df['bonus'] = np.where(df['ratio'] <= threshold_min, 0,
                       np.where(df['ratio'] <= threshold_max,
                                (df['ratio'] - threshold_min) / (threshold_max - threshold_min) * max_bonus,
                                max_bonus))
df['bonus'] = df['bonus'].round(0).astype(int)

# Display results
print("\n### Predictions and Bonus ###")
print(df[['mm-yyyy', 'total_dinero', 'expected_total_dinero', 'bonus']].to_string(index=False))