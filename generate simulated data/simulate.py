import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_data(num_months=120, start_date='2024-07-01'):
    """
    Generate synthetic sales data using Monte Carlo simulation with trend and seasonality.
    
    Parameters:
    - num_months: Number of months of data to generate (default 120 for 10 years)
    - start_date: Starting date for the data
    
    Returns:
    - DataFrame with synthetic sales data
    """
    # Initialize random seed for reproducibility
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range(start=start_date, periods=num_months, freq='M')
    
    # Base values and standard deviations based on existing data
    base_values = {
        'total_cajas': 4000,
        'total_dinero': 8000000,
        'cantidad_de_ordenes': 190,
        'ofertas_aplicadas': 220,
        'sku_activados': 85,
        'cantidad_facturas_venta': 180,
        'clientes_activados': 92,
        'total_clientes': 108,
        'total_sku_portafolio': 215
    }
    
    # Annual growth rates (as percentages)
    growth_rates = {
        'total_cajas': 0.05,      # 5% annual growth
        'total_dinero': 0.08,     # 8% annual growth
        'cantidad_de_ordenes': 0.03,  # 3% annual growth
        'ofertas_aplicadas': 0.04,    # 4% annual growth
        'sku_activados': 0.02,    # 2% annual growth
        'cantidad_facturas_venta': 0.03,  # 3% annual growth
        'clientes_activados': 0.01,    # 1% annual growth
    }
    
    std_devs = {
        'total_cajas': 800,
        'total_dinero': 1500000,
        'cantidad_de_ordenes': 20,
        'ofertas_aplicadas': 40,
        'sku_activados': 10,
        'cantidad_facturas_venta': 25,
        'clientes_activados': 5,
    }
    
    # Generate data
    data = []
    for i in range(num_months):
        # Calculate trend factor (monthly growth rate)
        months_passed = i
        trend_factors = {
            key: (1 + rate/12) ** months_passed 
            for key, rate in growth_rates.items()
        }
        
        # Add seasonal variation
        seasonal_factor = 1.0
        if dates[i].month == 12:  # December
            seasonal_factor = 1.3
        elif dates[i].month in [1, 2]:  # January and February
            seasonal_factor = 0.9
        elif dates[i].month in [6, 7]:  # June and July (summer months)
            seasonal_factor = 1.1
            
        # Generate random values with normal distribution and trend
        row = {
            'mm-yyyy': dates[i].strftime('%m-%Y'),
            'ruta_venta': 'RUTA 01',
            'total_cajas': max(0, np.random.normal(
                base_values['total_cajas'] * trend_factors['total_cajas'] * seasonal_factor,
                std_devs['total_cajas'] * trend_factors['total_cajas']
            )),
            'total_dinero': max(0, np.random.normal(
                base_values['total_dinero'] * trend_factors['total_dinero'] * seasonal_factor,
                std_devs['total_dinero'] * trend_factors['total_dinero']
            )),
            'cantidad_de_ordenes': max(0, np.random.normal(
                base_values['cantidad_de_ordenes'] * trend_factors['cantidad_de_ordenes'] * seasonal_factor,
                std_devs['cantidad_de_ordenes'] * trend_factors['cantidad_de_ordenes']
            )),
            'ofertas_aplicadas': max(0, np.random.normal(
                base_values['ofertas_aplicadas'] * trend_factors['ofertas_aplicadas'] * seasonal_factor,
                std_devs['ofertas_aplicadas'] * trend_factors['ofertas_aplicadas']
            )),
            'sku_activados': max(0, np.random.normal(
                base_values['sku_activados'] * trend_factors['sku_activados'] * seasonal_factor,
                std_devs['sku_activados'] * trend_factors['sku_activados']
            )),
            'cantidad_facturas_venta': max(0, np.random.normal(
                base_values['cantidad_facturas_venta'] * trend_factors['cantidad_facturas_venta'] * seasonal_factor,
                std_devs['cantidad_facturas_venta'] * trend_factors['cantidad_facturas_venta']
            )),
            'clientes_activados': max(0, np.random.normal(
                base_values['clientes_activados'] * trend_factors['clientes_activados'] * seasonal_factor,
                std_devs['clientes_activados'] * trend_factors['clientes_activados']
            )),
            'total_clientes': base_values['total_clientes'],
            'total_sku_portafolio': base_values['total_sku_portafolio']
        }
        
        # Calculate derived metrics
        row['sku_x_factura'] = row['sku_activados'] / row['cantidad_facturas_venta'] if row['cantidad_facturas_venta'] > 0 else 0
        row['ros_factura'] = row['total_dinero'] / row['cantidad_facturas_venta'] if row['cantidad_facturas_venta'] > 0 else 0
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Format numbers
    df['total_cajas'] = df['total_cajas'].round(2)
    df['total_dinero'] = df['total_dinero'].round(2)
    df['cantidad_de_ordenes'] = df['cantidad_de_ordenes'].round(0)
    df['ofertas_aplicadas'] = df['ofertas_aplicadas'].round(0)
    df['sku_activados'] = df['sku_activados'].round(0)
    df['cantidad_facturas_venta'] = df['cantidad_facturas_venta'].round(0)
    df['sku_x_factura'] = df['sku_x_factura'].round(2)
    df['ros_factura'] = df['ros_factura'].round(2)
    df['clientes_activados'] = df['clientes_activados'].round(0)
    
    return df

if __name__ == "__main__":
    # Generate 10 years (120 months) of synthetic data
    synthetic_data = generate_synthetic_data(num_months=120)
    
    # Save to CSV
    synthetic_data.to_csv('data/simulated/synthetic_sales_data_10years.csv', index=False)
    print("10 years of synthetic data generated and saved to 'data/simulated/synthetic_sales_data_10years.csv'")
