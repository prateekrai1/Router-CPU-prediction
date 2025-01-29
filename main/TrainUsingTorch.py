import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
train_df = pd.read_csv(r"E:\Router CPU prediction\data\train.csv", parse_dates=['timestamp'], index_col='timestamp')
test_df = pd.read_csv(r"E:\Router CPU prediction\data\test.csv", parse_dates=['timestamp'], index_col='timestamp')

# Normalize CPU usage values
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

# Convert to tensors
train_data = torch.tensor(train_scaled, dtype=torch.float32).squeeze()  # Ensure 1D tensor
test_data = torch.tensor(test_scaled, dtype=torch.float32).squeeze()    # Ensure 1D tensor
print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)
# Function to create sequences for RNN
def create_sequences(data, time_steps=5):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])  # Take a sequence of 'time_steps' CPU usage values
        y.append(data[i+time_steps])    # Target is the next CPU usage value
    X = torch.stack(X)  # Convert list of tensors to a single tensor
    y = torch.stack(y)  # Convert list of tensors to a single tensor
    return X, y

# Create sequences
time_steps = 5
X_train, y_train = create_sequences(train_data, time_steps)
X_test, y_test = create_sequences(test_data, time_steps)
# Reshape y_train and y_test to ensure the shape matches the model output
y_train = y_train.view(-1, 1)
y_test = y_test.view(-1, 1)
print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

# Define RNN model
class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Take last time step output
        return out

# Initialize model
model = RNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 50
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train.unsqueeze(-1))  # Add feature dimension
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Test the model
model.eval()
y_pred = model(X_test.unsqueeze(-1)).detach().numpy()  # Add feature dimension

# Inverse transform predictions
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

# Plot Actual vs. Predicted CPU Usage
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label="Actual CPU Usage", color='blue')
plt.plot(y_pred_inv, label="Predicted CPU Usage", linestyle="dashed", color='red')
plt.legend()
plt.xlabel("Time Steps")
plt.ylabel("CPU Usage")
plt.title("Actual vs. Predicted CPU Usage")
plt.show()