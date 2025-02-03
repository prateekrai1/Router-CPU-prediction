import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import r2_score


# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional: Makes things slower, but more reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loading and preprocessing
# data = pd.read_csv('output.csv')
data = pd.read_csv('./merged_cpu_bandwidth.csv')
data['CPU Usage'] = data['CPU Usage'].replace('%', '', regex=True)
data['CPU Usage'] = pd.to_numeric(data['CPU Usage'], errors='coerce')

# Handle missing values in numeric columns by replacing them with the mean
numeric_columns = data.select_dtypes(include=['number']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Handle missing values in datetime columns by coercing invalid entries and filling with the mode
datetime_columns = data.select_dtypes(include=['datetime']).columns
for col in datetime_columns:
    data[col] = pd.to_datetime(data[col], errors='coerce')  # Convert to datetime and handle errors
    data[col].fillna(data[col].mode()[0], inplace=True)  # Fill missing with the mode (most frequent value)
selected_features = ['CPU Usage', 'Used Memory (KB)', 'TCP Count', 'UDP Count', 'Bandwidth (bps)']
scaler = MinMaxScaler(feature_range=(0, 1))


scaled_data = scaler.fit_transform(data[selected_features].values)
scaled_df = pd.DataFrame(scaled_data, columns=selected_features)

# Convert data to sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        x = data.iloc[i:(i+seq_length)].values  # Use all features for inputs
        y = data.iloc[i+seq_length, 0]  # Target is cpu_usage
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 8  # Sequence length
X, y = create_sequences(scaled_df, seq_length)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)




class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Define the fully connected layer that maps from hidden state space to output space
        self.fc = nn.Linear(hidden_size, output_size)
        # Optional: Define an activation function if needed
        # self.relu = nn.Tanh()

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Apply the linear layer and then ReLU activation function
        out = self.fc(out[:, -1, :])
        # out = self.relu(out)  # Apply ReLU activation function
        return out





# Initialize model, loss criterion, and optimizer
model = LSTMModel(input_size=X_train.shape[2], hidden_size=50, num_layers=2, output_size=1).to(device)


criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training the model
loss_values = []

num_epochs = 50
for epoch in range(num_epochs):
    for features, labels in train_loader:
        outputs = model(features)
        loss = criterion(outputs, labels.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # running_loss += loss.item() * inputs.size(0)

    # epoch_loss = running_loss / len(train_loader.dataset)
    loss_values.append(loss.item())
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Continue from the existing code
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test.unsqueeze(1))
    r2 = r2_score(y_test.cpu().numpy(), predictions.cpu().numpy())  # Compute R-squared score


# Convert predictions and actual values to numpy for plotting
predicted_cpu_usage = predictions.cpu().numpy()
actual_cpu_usage = y_test.cpu().numpy()


predicted_cpu_usage = np.array(predicted_cpu_usage).reshape(-1, 1)
dummy_array = np.zeros((predicted_cpu_usage.shape[0], 4))  # Creating dummy array for features
full_input = np.hstack((predicted_cpu_usage, dummy_array))  # Combine predicted CPU usage with dummy array
predicted_cpu_usage = scaler.inverse_transform(full_input)[:, 0]  # Inverse transform and extract CPU Usage

actual_cpu_usage = actual_cpu_usage.reshape(-1, 1)  # Ensure actual CPU usage is 2D
full_input_actual = np.hstack((actual_cpu_usage, dummy_array))  # Combine actual with dummy array
actual_cpu_usage = scaler.inverse_transform(full_input_actual)[:, 0]  # Inverse transform and extract CPU Usage
print('predicted_cpu_usage: ', predicted_cpu_usage[:10])
print('actual_cpu_usage: ', actual_cpu_usage[:10])
print(f'Test MSE: {test_loss.item()}')
print(f'Test R-squared: {r2}')  # Display the R-squared score

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(actual_cpu_usage, label='Actual CPU Usage', marker='o')
plt.plot(predicted_cpu_usage, label='Predicted CPU Usage', linestyle='--', marker='x')
plt.title('Comparison of Actual and Predicted CPU Usage of Complicated Model')
plt.xlabel('Test Samples')
plt.ylabel('CPU Usage (scaled)')
plt.legend()
plt.show()

