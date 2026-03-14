import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

from attention_predictor import AttentionLSTM, create_sequences, window_based_evaluation, process_optical_data
from cisco_predictor import process_cisco_data

def train_evaluate_merged():
    print("Loading Datasets for Merged Omni-Architecture...")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # ---------------- 1. Load Optical Data ----------------
    opt_train_path = os.path.join(base_dir, 'telemetry/optical-failure-dataset/SoftFailure_dataset.csv')
    opt_test_path = os.path.join(base_dir, 'telemetry/optical-failure-dataset/HardFailure_dataset.csv')
    X_opt_train_raw, y_opt_train_raw, opt_feature_cols = process_optical_data(opt_train_path)
    X_opt_test_raw, y_opt_test_raw, _ = process_optical_data(opt_test_path)
    
    scaler_opt = StandardScaler()
    X_opt_train_scaled = scaler_opt.fit_transform(X_opt_train_raw)
    X_opt_test_scaled = scaler_opt.transform(X_opt_test_raw)
    
    # ---------------- 2. Load Cisco Data ----------------
    cisco_data = os.path.join(base_dir, 'telemetry/cisco-telemetry/2/bgpclear.csv')
    cisco_truth = os.path.join(base_dir, 'telemetry/cisco-telemetry/2/bgpclear_ground_truth.txt')
    X_cis_train_raw, y_cis_train_raw, X_cis_test_raw, y_cis_test_raw, cis_feature_cols = process_cisco_data(cisco_data, cisco_truth)
    
    scaler_cis = StandardScaler()
    X_cis_train_scaled = scaler_cis.fit_transform(X_cis_train_raw)
    X_cis_test_scaled = scaler_cis.transform(X_cis_test_raw)
    
    # ---------------- 3. SMOTE Balancing ----------------
    print("Applying SMOTE to Optical Dataset...")
    smote_opt = SMOTE(sampling_strategy=0.3, random_state=42)
    X_opt_train_res, y_opt_train_res = smote_opt.fit_resample(X_opt_train_scaled, y_opt_train_raw)
    
    print("Applying SMOTE to Cisco Dataset...")
    smote_cis = SMOTE(sampling_strategy=0.3, random_state=42)
    X_cis_train_res, y_cis_train_res = smote_cis.fit_resample(X_cis_train_scaled, y_cis_train_raw)
    
    # ---------------- 4. Sequence Generation ----------------
    seq_length = 15
    X_opt_train_seq, y_opt_train_seq = create_sequences(X_opt_train_res, y_opt_train_res, seq_length)
    X_opt_test_seq, y_opt_test_seq = create_sequences(X_opt_test_scaled, y_opt_test_raw, seq_length)
    
    X_cis_train_seq, y_cis_train_seq = create_sequences(X_cis_train_res, y_cis_train_res, seq_length)
    X_cis_test_seq, y_cis_test_seq = create_sequences(X_cis_test_scaled, y_cis_test_raw, seq_length)
    
    # ---------------- 5. Feature Padding & Merging ----------------
    # Optical has ~12 features, Cisco has 8. We pad them together so the model can read both simultaneously
    f_opt = X_opt_train_seq.shape[2]
    f_cis = X_cis_train_seq.shape[2]
    total_features = f_opt + f_cis
    print(f"Consolidating feature spaces: Optical ({f_opt}) + Cisco ({f_cis}) = Total Features: {total_features}")
    
    def pad_optical(X):
        # Pad cisco features with 0 for optical datasets
        return np.concatenate([X, np.zeros((X.shape[0], X.shape[1], f_cis))], axis=2)
        
    def pad_cisco(X):
        # Pad optical features with 0 for cisco datasets
        return np.concatenate([np.zeros((X.shape[0], X.shape[1], f_opt)), X], axis=2)
        
    X_train_merged = np.concatenate([pad_optical(X_opt_train_seq), pad_cisco(X_cis_train_seq)], axis=0)
    y_train_merged = np.concatenate([y_opt_train_seq, y_cis_train_seq], axis=0)
    
    X_test_merged = np.concatenate([pad_optical(X_opt_test_seq), pad_cisco(X_cis_test_seq)], axis=0)
    y_test_merged = np.concatenate([y_opt_test_seq, y_cis_test_seq], axis=0)
    
    print(f"Merged Omni-Training Sequences: {len(X_train_merged)} (Failures: {int(y_train_merged.sum())})")
    print(f"Merged Omni-Testing Sequences: {len(X_test_merged)} (Failures: {int(y_test_merged.sum())})")
    
    # ---------------- 6. PyTorch Training Loop ----------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = TensorDataset(torch.tensor(X_train_merged, dtype=torch.float32), torch.tensor(y_train_merged, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test_merged, dtype=torch.float32), torch.tensor(y_test_merged, dtype=torch.float32))
    
    # High batch size for rapid learning across hundreds of thousands of merged rows
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)
    
    model = AttentionLSTM(input_size=total_features).to(device)
    
    num_neg = (y_train_merged == 0).sum()
    num_pos = (y_train_merged == 1).sum()
    pos_weight_val = num_neg / num_pos if num_pos > 0 else 1.0 
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    epochs = 6
    print("\\nTraining Merged Omni-Attention-LSTM...")
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
            
    # ---------------- 7. Dual Evaluation ----------------
    model.eval()
    lstm_preds = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = torch.sigmoid(model(batch_X)).squeeze()
            preds = (outputs > 0.4).int().cpu().numpy()
            if preds.ndim == 0:
                lstm_preds.append(preds.item())
            else:
                lstm_preds.extend(preds)
                
    lstm_preds = np.array(lstm_preds)
    
    standard_f1 = f1_score(y_test_merged, lstm_preds, zero_division=0)
    print(f"\\n--- Omni Hardware-Agnostic Standard F1-Score: {standard_f1:.4f} ---")
    
    win_prec, win_rec, win_f1 = window_based_evaluation(y_test_merged, lstm_preds, window_size=17)
    print(f"\\n--- Omni Hardware-Agnostic 60-second Window F1-Score: {win_f1:.4f} ---")

if __name__ == '__main__':
    train_evaluate_merged()
