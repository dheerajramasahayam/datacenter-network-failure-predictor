import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import AttentionLSTM, create_sequences
from data.preprocessing import process_telemetry_data

def evaluate_model():
    print("Initializing Attention-LSTM Inference Pipeline...")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 1. Load Data
    train_path = os.path.join(base_dir, 'telemetry/optical-failure-dataset/SoftFailure_dataset.csv')
    test_path = os.path.join(base_dir, 'telemetry/optical-failure-dataset/HardFailure_dataset.csv')
    
    X_train_raw, _, feature_cols = process_telemetry_data(train_path)
    X_test_raw, y_test_raw, _ = process_telemetry_data(test_path)
    
    # 2. Scale (fit on training only)
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # 3. Create Sequences
    seq_length = 15
    X_test, y_test = create_sequences(X_test_scaled, y_test_raw, seq_length)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(base_dir, 'models/attention_lstm.pth')
    
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}. Run train.py first.")
        return
        
    model = AttentionLSTM(input_size=len(feature_cols)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 4. Inference
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = torch.sigmoid(model(X_test_tensor)).squeeze()
        preds = (outputs > 0.5).int().cpu().numpy()
        
    # 5. Evaluation
    standard_f1 = f1_score(y_test, preds, zero_division=0)
    print(f"\\nInference Complete.")
    print(f"Zero-Shot Evaluation F1-Score: {standard_f1:.4f}")

if __name__ == '__main__':
    evaluate_model()
