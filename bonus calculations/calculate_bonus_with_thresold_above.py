import pandas as pd
import numpy as np

# Load your DataFrame with actual and expected revenue
df = pd.read_csv('data/ruta_01_with_expected_ridge_regression.csv')

# Calculate the ratio of actual to expected revenue
df['ratio'] = df['total_dinero'] / df['expected_total_dinero']

# Define bonus parameters
threshold = 1.015  # New threshold for faster bonus increase
max_bonus = 50000

# Calculate the bonus
df['bonus'] = np.where(df['ratio'] <= 1, 0,
                       np.where(df['ratio'] <= threshold, 
                                (df['ratio'] - 1) / (threshold - 1) * max_bonus, 
                                max_bonus))
df['bonus'] = df['bonus'].round(0).astype(int)

# Display results
print("\n### Predictions and Bonus ###")
print(df[['mm-yyyy', 'total_dinero', 'expected_total_dinero', 'bonus']].to_string(index=False))

# Optional: Save the updated DataFrame
# df.to_csv('data/ruta_01_with_bonus.csv', index=False)