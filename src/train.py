import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import AttentionLSTM, create_sequences
from data.preprocessing import process_telemetry_data

def train_model():
    print("Initializing Attention-LSTM Training Pipeline...")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 1. Load Data
    train_path = os.path.join(base_dir, 'telemetry/optical-failure-dataset/SoftFailure_dataset.csv')
    X_train_raw, y_train_raw, feature_cols = process_telemetry_data(train_path)
    
    # 2. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    
    # 3. Create Sequences
    seq_length = 15
    X_train, y_train = create_sequences(X_train_scaled, y_train_raw, seq_length)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}")
    
    # 4. Data Loaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    # 5. Model Initialization
    model = AttentionLSTM(input_size=len(feature_cols)).to(device)
    
    # BCE Loss weighted for anomaly imbalance
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    pos_weight_val = num_neg / num_pos if num_pos > 0 else 1.0 
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # 6. Training Loop
    epochs = 12
    print("\\n--- Commencing PyTorch Optimization ---")
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
            
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}")
            
    # 7. Save Artifacts
    os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(base_dir, 'models/attention_lstm.pth'))
    print("Model compilation finished. Saved to models/attention_lstm.pth.")

if __name__ == '__main__':
    train_model()
