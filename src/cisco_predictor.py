import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from attention_predictor import AttentionLSTM, create_sequences, window_based_evaluation

def process_cisco_data(data_path, truth_path):
    print("Loading Cisco BGP Clear Dataset...")
    # Read specific columns to save memory
    cols = ['time', 'Producer', 'bytes-received', 'bytes-sent', 'input-data-rate', 'input-drops', 'input-errors', 'output-data-rate', 'output-drops', 'output-errors']
    # The dataset has malformed rows; we instruct pandas to skip bad lines
    df = pd.read_csv(data_path, usecols=cols, on_bad_lines='skip', engine='python')
    
    # The timestamps in Cisco dataset are in nanoseconds, some headers might be embedded as strings
    df['time'] = pd.to_numeric(df['time'], errors='coerce') / 1e9
    
    print("Applying Ground Truth Failure Labels...")
    truth = pd.read_csv(truth_path)
    df['target'] = 0
    
    # Tag failures
    for _, row in truth.iterrows():
        node = row['Node']
        start = row['Start']
        end = row['End']
        mask = (df['Producer'] == node) & (df['time'] >= start) & (df['time'] <= end)
        df.loc[mask, 'target'] = 1
        
    print(f"Total rows: {len(df)}, Total Outage rows: {df['target'].sum()}")
    
    # Sort chronologically to maintain time-series integrity
    df = df.sort_values('time').reset_index(drop=True)
    
    # Forward fill to handle any sporadic NaNs in telemetry streams
    df = df.ffill().bfill()
    
    # Select feature columns
    feature_cols = [c for c in cols if c not in ['time', 'Producer']]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
    # Split chronologically: First 60% for training (Soft Failures / Early anomalies), Last 40% for testing (Zero-shot generalization)
    split_idx = int(len(df) * 0.6)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    return train_df[feature_cols].values, train_df['target'].values, test_df[feature_cols].values, test_df['target'].values, feature_cols

def train_evaluate_cisco():
    print("Starting Cisco Telemetry Cross-Validation...")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, 'telemetry/cisco-telemetry/2/bgpclear.csv')
    truth_path = os.path.join(base_dir, 'telemetry/cisco-telemetry/2/bgpclear_ground_truth.txt')
    
    X_train_raw, y_train_raw, X_test_raw, y_test_raw, feature_cols = process_cisco_data(data_path, truth_path)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    seq_length = 15
    
    print("Applying SMOTE to balance the training sequences...")
    # SMOTE operates on 2D data, so we apply it before sequence generation
    smote = SMOTE(sampling_strategy=0.3, random_state=42) # Bring minority class up to 30%
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train_raw)
    
    X_train, y_train = create_sequences(X_train_res, y_train_res, seq_length)
    X_test, y_test = create_sequences(X_test_scaled, y_test_raw, seq_length)
    
    print(f"Cisco Attention-LSTM Training Sequences: {len(X_train)} (Failures: {int(y_train.sum())})")
    print(f"Cisco Attention-LSTM Testing Sequences: {len(X_test)} (Failures: {int(y_test.sum())})")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True) # Larger batch size for the 700k row dataset
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
    
    model = AttentionLSTM(input_size=len(feature_cols)).to(device)
    
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    pos_weight_val = num_neg / num_pos if num_pos > 0 else 1.0 
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    epochs = 6 # Increased epochs
    print("\\nTraining Cisco Attention-LSTM...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            if outputs.ndim == 0: outputs = outputs.unsqueeze(0)
            if batch_y.ndim == 0: batch_y = batch_y.unsqueeze(0)
                
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
            
    # Evaluation
    model.eval()
    lstm_preds = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = torch.sigmoid(model(batch_X)).squeeze()
            # Lowering the rigorous threshold to 0.4 for heavily imbalanced anomaly detection
            preds = (outputs > 0.4).int().cpu().numpy()
            if preds.ndim == 0:
                lstm_preds.append(preds.item())
            else:
                lstm_preds.extend(preds)
                
    lstm_preds = np.array(lstm_preds)
    
    standard_f1 = f1_score(y_test, lstm_preds, zero_division=0)
    print(f"\\n--- Cisco Attention-LSTM Standard F1-Score: {standard_f1:.4f} ---")
    
    win_prec, win_rec, win_f1 = window_based_evaluation(y_test, lstm_preds, window_size=17)
    print(f"\\n--- Cisco Attention-LSTM 60-second Window F1-Score: {win_f1:.4f} ---")

if __name__ == '__main__':
    train_evaluate_cisco()
