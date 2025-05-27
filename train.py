import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
# from scales import LSTMAutoencoder  

x_train_tensor = torch.randn(9462, 200, 6)  # (num_sequences, sequence_length, num_features)

# Model inference
# Define input_size and hidden_size 
input_size = x_train_tensor.shape[2]  # Number of features per time step
hidden_size = 64  

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True) #defines an encoder to compress input
        self.decoder = nn.LSTM(hidden_size, input_size, batch_first=True) #define decoder to reconstruct the originl sequence 

    def forward(self, x):
        _, (h_n, _) = self.encoder(x)
        h_n = h_n.repeat(x.size(1), 1, 1).transpose(0, 1)
        decoded, _ = self.decoder(h_n)
        return decoded

# Compute reconstruction error for all training sequences
model = LSTMAutoencoder(input_size, hidden_size)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
reconstruction_errors = []

with torch.no_grad():
    for i, sequence in enumerate(x_train_tensor):
        sequence = sequence.unsqueeze(0)  # Add batch dimension
        reconstructed = model(sequence)
        mse = nn.functional.mse_loss(reconstructed, sequence, reduction='mean').item()
        reconstruction_errors.append(mse)

# Convert to numpy array
reconstruction_errors = np.array(reconstruction_errors)

# Compute threshold using k-mean deviation (mean + k * std)
k = 3  #3 for strict(detects only highly abnormal values), could be 2 or 1, for lenient model(would result in more false positives)
mean_error = np.mean(reconstruction_errors)
std_error = np.std(reconstruction_errors)
threshold = mean_error + k * std_error #sequences with error above this are treated as anomalies

print(f"\nAnomaly Threshold (K-Means Deviation): {threshold:.6f}")
# Optionally, print anomalies for all sequences 
for i, mse in enumerate(reconstruction_errors):
    if mse > threshold:
        print(f"⚠️ ALERT! Anomaly detected at index {i} | MSE: {mse:.6f}")

#'------------------------------------------------------------------------------------------------------------------------------------'

# Plot reconstruction errors and threshold
plt.figure(figsize=(12, 6))
plt.plot(reconstruction_errors, label='Reconstruction Error')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Errors & Anomaly Threshold')
plt.xlabel('Sequence Index')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show() 