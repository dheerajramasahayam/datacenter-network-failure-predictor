import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Calculate attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        # Context vector
        context = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.fc(context)
        return out

def create_sequences(data, targets, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = targets[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def process_optical_data(data_path):
    df = pd.read_csv(data_path)
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df['target'] = df['Failure'].notna().astype(int)
    df['BER'] = pd.to_numeric(df['BER'], errors='coerce')
    df['OSNR'] = pd.to_numeric(df['OSNR'], errors='coerce')
    df['InputPower'] = pd.to_numeric(df['InputPower'], errors='coerce')
    df['OutputPower'] = pd.to_numeric(df['OutputPower'], errors='coerce')
    df = df.ffill().bfill()
    df_encoded = pd.get_dummies(df, columns=['Type', 'ID'])
    feature_cols = ['BER', 'OSNR', 'InputPower', 'OutputPower'] + [c for c in df_encoded.columns if c.startswith('Type_') or c.startswith('ID_')]
    return df_encoded[feature_cols].values, df_encoded['target'].values, feature_cols

def window_based_evaluation(y_true, y_pred, window_size=17): # ~60 seconds (each step 3.5s -> 17 steps = 59.5s)
    # If the model predicts 1 within window_size steps prior to an actual 1, it's a True Valid Prediction
    y_true_windowed = np.zeros_like(y_true)
    
    # Expand true anomalies backwards
    for i in range(len(y_true)):
        if y_true[i] == 1:
            start = max(0, i - window_size)
            y_true_windowed[start:i+1] = 1
            
    # Calculate True Positives: Predicted 1 and Ground truth is inside the expanded window
    tp = np.sum((y_pred == 1) & (y_true_windowed == 1))
    # False Positives: Predicted 1 but totally outside window
    fp = np.sum((y_pred == 1) & (y_true_windowed == 0))
    
    # Calculate False Negatives (Missed the outage entirely within the whole window leading up to it)
    fn_count = 0
    in_failure = False
    predicted_in_failure = False
    
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_failure:
            in_failure = True
            predicted_in_failure = False
            start = max(0, i - window_size)
            if np.any(y_pred[start:i+1] == 1):
                predicted_in_failure = True
                
        elif y_true[i] == 1 and in_failure:
            if not predicted_in_failure:
                start = max(0, i - window_size)
                if np.any(y_pred[start:i+1] == 1):
                    predicted_in_failure = True
                    
        elif y_true[i] == 0 and in_failure:
            in_failure = False
            if not predicted_in_failure:
                fn_count += 1
                
    if in_failure and not predicted_in_failure:
        fn_count += 1
        
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn_count) if (tp + fn_count) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def train_evaluate_attention():
    print("Loading Real-World Optical Telemetry for Attention-LSTM...")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    train_path = os.path.join(base_dir, 'telemetry/optical-failure-dataset/SoftFailure_dataset.csv')
    test_path = os.path.join(base_dir, 'telemetry/optical-failure-dataset/HardFailure_dataset.csv')
    
    X_train_raw, y_train_raw, feature_cols = process_optical_data(train_path)
    X_test_raw, y_test_raw, _ = process_optical_data(test_path)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    seq_length = 15
    X_train, y_train = create_sequences(X_train_scaled, y_train_raw, seq_length)
    X_test, y_test = create_sequences(X_test_scaled, y_test_raw, seq_length)
    
    print(f"Attention-LSTM Training Sequences (Soft Failures): {len(X_train)} (Failures: {y_train.sum()})")
    print(f"Attention-LSTM Testing Sequences (Hard Failures): {len(X_test)} (Failures: {y_test.sum()})")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    model = AttentionLSTM(input_size=len(feature_cols)).to(device)
    
    # BCE Loss with positive weight
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    pos_weight_val = num_neg / num_pos if num_pos > 0 else 1.0 
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    epochs = 12
    print("\\nTraining Attention-LSTM...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            
            # Handle edge case where batch_size=1
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
            if batch_y.ndim == 0:
                batch_y = batch_y.unsqueeze(0)
                
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
            preds = (outputs > 0.5).int().cpu().numpy()
            if preds.ndim == 0:
                lstm_preds.append(preds.item())
            else:
                lstm_preds.extend(preds)
                
    lstm_preds = np.array(lstm_preds)
    
    # Classical Metric evaluation
    standard_precision = precision_score(y_test, lstm_preds, zero_division=0)
    standard_recall = recall_score(y_test, lstm_preds, zero_division=0)
    standard_f1 = f1_score(y_test, lstm_preds, zero_division=0)
    print(f"\\n--- Attention-LSTM Standard F1-Score: {standard_f1:.4f} ---")
    
    # Academic Window-based Evaluation
    win_prec, win_rec, win_f1 = window_based_evaluation(y_test, lstm_preds, window_size=17) # 17 * 3.5s ~= 60s
    print(f"\\n--- Attention-LSTM 60-second Window F1-Score: {win_f1:.4f} ---")
    
    # Save output for research paper
    os.makedirs(os.path.join(base_dir, 'results'), exist_ok=True)
    with open(os.path.join(base_dir, 'results/attention_lstm_report.md'), 'w') as f:
        f.write("# Research Output: Attention-LSTM with Windowed Evaluation\\n\\n")
        f.write("Validation Performed on: Network-And-Services/optical-failure-dataset\\n\\n")
        f.write("## 1. Classical Static Evaluation (Exact Predict Match)\\n")
        f.write(f"- Precision: {standard_precision:.4f}\\n")
        f.write(f"- Recall: {standard_recall:.4f}\\n")
        f.write(f"- F1-Score: {standard_f1:.4f}\\n\\n")
        f.write("## 2. Realistic 60-Second Window Evaluation\\n")
        f.write(f"- Precision: {win_prec:.4f}\\n")
        f.write(f"- Recall: {win_rec:.4f}\\n")
        f.write(f"- F1-Score: {win_f1:.4f}\\n\\n")
    
    print("Report written to results/attention_lstm_report.md")

if __name__ == '__main__':
    train_evaluate_attention()
