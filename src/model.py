import torch
import torch.nn as nn
import numpy as np

class AttentionLSTM(nn.Module):
    """
    Temporal Attention-Guided Sequence Learning Model 
    for Zero-Shot Generalization in Datacenter Network Failures.
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(AttentionLSTM, self).__init__()
        # PyTorch sequence processor
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        
        # Linear Self-Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Dense classification layer
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Calculate dynamic attention weights for every microsecond in the sequence window
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Condense the context vector against the weighted attention matrix
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Feed-forward
        out = self.fc(context)
        return out

def create_sequences(data, targets, seq_length):
    """
    Convert continuous temporal arrays into rolling sliding-windows 
    for the PyTorch LSTM consumption.
    """
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = targets[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
