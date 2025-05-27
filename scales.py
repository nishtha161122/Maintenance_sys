import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

df = pd.read_csv('datetime.csv')
sensor_columns = [
    'air_temperature', 'process_temperature', 'rotational_speed', 
    'Torque', 'tool_wear', 'machine_failure'
]
sensor_data = df[sensor_columns]

#scaling data to normalized form (0-1 range; using fit transform)
scaler = MinMaxScaler()
sensor_data_scaled = scaler.fit_transform(sensor_data)
scaled_df = pd.DataFrame(sensor_data_scaled, columns=sensor_columns)

#inserting date column at begining 
scaled_df.insert(0, 'date', df['date'])
scaled_df['machine_failure'] = df['machine_failure'] #overwrites scaled machine-failure with original binary values(0/1)

sensor_data_normal = scaled_df[scaled_df['machine_failure']==0].reset_index(drop=True) #no machine failure data

#saving data with no machine fialure in non failure
sensor_data_normal.to_csv('nonfailure.csv', index=False)

#display few rows form filtered data
print("Original data shape:", scaled_df.shape)
print("Normal operations data shape:", sensor_data_normal.shape)
print("\nFirst few rows of normal operations data:")
print(sensor_data_normal.head())
#'------------------------------------------------------------------------------------------------------------------------------------'

# First, select only numerical columns for sequence creation
numerical_columns = [
    'air_temperature', 'process_temperature', 'rotational_speed', 
    'Torque', 'tool_wear', 'machine_failure'
]

# Get numerical data only
numerical_data = sensor_data_normal[numerical_columns].values

# Create sequences from numerical data
def create_sequences(data, time_steps=200): 
    sequences = []
    for i in range(len(data) - time_steps + 1):
        sequences.append(data[i:(i + time_steps)])
    return np.array(sequences, dtype=np.float32)  # Specify dtype here

# Create sequences with numerical data only
x_train = create_sequences(numerical_data, time_steps=200)
print("Sequence shape:", x_train.shape)

#'------------------------------------------------------------------------------------------------------------------------------------'

# Convert data to pytorch tensor for training
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)

# Define LSTM Autoencoder
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

# Model initialization
input_size = x_train_tensor.shape[2]  
hidden_size = 64
model = LSTMAutoencoder(input_size, hidden_size) 

# Choose one sequence to test
original_sequence = x_train_tensor[0].unsqueeze(0)  

# Prepare DataLoader
train_dataset = TensorDataset(x_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in train_loader:
        batch_x = batch[0]
        output = model(batch_x)
        loss = criterion(output, batch_x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
torch.save(model, 'model_weights.pth')

#------------------------------------------------------------------------------------------------------------------------------------
