# AI Model for Salesman Compensation

This project implements various machine learning models to predict and analyze salesman compensation based on sales performance data. The system includes multiple models and data processing pipelines to handle different aspects of sales compensation analysis.

## Project Structure

- `data/`: Contains all data files
  - `raw_data.csv`: Original sales data
  - `raw_data_ruta_01.csv`: Specific route data
  - Various model-specific prediction files
  - `simulated/`: Contains generated test data

- `model/`: Contains trained model files
  - `sarima_model.pkl`: SARIMA time series model
  - `random_forest_model.pkl`: Random Forest regression model
  - `custom_linear_model.pkl`: Custom linear regression model
  - `ridge_model.pkl`: Ridge regression model

- `build model/`: Scripts for building and training models
- `generate simulated data/`: Tools for generating test data
- `clean data/`: Data preprocessing and cleaning scripts
- `bonus calculations/`: Compensation calculation utilities

## Setup and Installation

1. Ensure you have Python 3.12 installed
2. Install dependencies using pipenv:
   ```bash
   pipenv install
   ```

## Dependencies

The project requires the following Python packages (specified in `requirements.txt`):


To install all dependencies:
```bash
pip install -r requirements.txt
```

## Data Usage

The project uses sales performance data with the following characteristics:
- Historical sales data
- Route-specific information
- Performance metrics
- Compensation calculations

## Running the Models

1. Data Preparation:
   - Place your raw data in the `data/` directory
   - Use the scripts in `clean data/` to preprocess your data

2. Model Training:
   - Navigate to the `build model/` directory
   - Run the appropriate model training script

3. Prediction:
   - Use the trained models to make predictions on new data
   - Results will be saved in the `data/` directory

## Model Types

1. SARIMA Model:
   - Time series analysis for sales patterns
   - Best for seasonal and trend analysis

2. Random Forest Model:
   - Handles complex non-linear relationships
   - Good for feature importance analysis

3. Custom Linear Model:
   - Simple linear regression with custom features
   - Easy to interpret results

4. Ridge Regression:
   - Regularized linear regression
   - Handles multicollinearity

## Output Files

The system generates prediction files in the `data/` directory:
- `ruta_01_with_expected_sarima.csv`
- `ruta_01_with_expected_custom_regression.csv`
- `ruta_01_with_expected_ridge_regression.csv`
- `ruta_01_with_expected_random_forest.csv`

## Notes

- All models are saved in pickle format for easy loading
- The system supports both real and simulated data
- Bonus calculations can be customized based on business rules 