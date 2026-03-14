import pandas as pd
import numpy as np

def process_telemetry_data(csv_path):
    """
    Standard Min-Max/Continuous temporal preprocessing engine 
    for Gigabit Optical metrics.
    """
    # Load raw telemetry
    df = pd.read_csv(csv_path)
    
    # Sort chronologically
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    # Map failure targets
    df['target'] = df['Failure'].notna().astype(int)
    
    # Force convert variable matrices to numeric floats
    for numeric_col in ['BER', 'OSNR', 'InputPower', 'OutputPower']:
        df[numeric_col] = pd.to_numeric(df[numeric_col], errors='coerce')
        
    # Forward and Backward fill telemetry latency drops
    df = df.ffill().bfill()
    
    # Categorical one-hot encoding for Network Transponder classes
    df_encoded = pd.get_dummies(df, columns=['Type', 'ID'])
    
    feature_cols = ['BER', 'OSNR', 'InputPower', 'OutputPower'] + \
                   [c for c in df_encoded.columns if c.startswith('Type_') or c.startswith('ID_')]
                   
    # Return feature arrays and boolean targets
    return df_encoded[feature_cols].values, df_encoded['target'].values, feature_cols
