


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



seed = 42
set_seed(seed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def extract_cpu_usage(filename):
#     cpu_usages = []

#     with open(filename, 'r') as file:
#         for line in file:
#             # Split the line by commas
#             parts = line.split(',')

#             # Find the part that contains 'CPU USAGE'
#             for part in parts:
#                 if 'CPU USAGE' in part:
#                     # Extract the percentage value
#                     usage_str = part.split(':')[1].strip()
#                     usage_value = int(usage_str.replace('%', ''))
#                     cpu_usages.append(usage_value)
#                     break  # We found the CPU usage, no need to check other parts

#     return cpu_usages

# def write_list_to_file(my_list, filename):
#     with open(filename, 'w') as file:
#         for item in my_list:
#             file.write(f"{item}\n")


# # Data loading and preprocessing
data = pd.read_csv('./merged_cpu_bandwidth.csv')
data['CPU Usage'] = data['CPU Usage'].replace('%', '', regex=True).astype(float)

numeric_columns = data.select_dtypes(include=['number']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

datetime_columns = data.select_dtypes(include=['datetime']).columns
for col in datetime_columns:
    data[col] = pd.to_datetime(data[col], errors='coerce')
    data[col].fillna(data[col].mode()[0], inplace=True)

selected_features = ['CPU Usage', 'Used Memory (KB)', 'TCP Count', 'UDP Count', 'Bandwidth (bps)']


def read_floats_from_file(filename):
    float_list = []
    with open(filename, 'r') as file:
        for line in file:
            # Convert each line to a float and add to the list
            float_list.append(float(line.strip()))
    return float_list



# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
# cpu_usages = scaler.fit_transform(cpu_usages)
# cpu_usages = scaler.fit_transform(cpu_usages.reshape(-1, 1))
data[selected_features] = scaler.fit_transform(data[selected_features])

# Convert data to sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)].values 
        y = data.iloc[i + seq_length]['CPU Usage'] 
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 8  # Sequence length
X, y = create_sequences(data[selected_features], seq_length)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


input_size = len(selected_features)
model = LSTMModel(input_size=input_size, hidden_size=50, num_layers=2, output_size=1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training the model
num_epochs = 50
loss_values = []
for epoch in range(num_epochs):
    for features, labels in train_loader:

        # print(features.shape)
        # assert 0 == 1
        # features = features.squeeze()
        outputs = model(features)
        loss = criterion(outputs, labels.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_values.append(loss.item())
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    X_test = X_test.squeeze()
    predictions = model(X_test)
    # predictions = model(X_test)
    test_loss = criterion(predictions, y_test.unsqueeze(1))
    r2 = r2_score(y_test.cpu().numpy(), predictions.cpu().numpy())

# Convert predictions and actual values to numpy
predicted_cpu_usage = predictions.cpu().numpy()
actual_cpu_usage = y_test.cpu().numpy()

# Inverse transform to get actual values
predicted_cpu_usage = np.array(predicted_cpu_usage).reshape(-1, 1)
dummy_array = np.zeros((predicted_cpu_usage.shape[0], 4))
full_input = np.hstack((predicted_cpu_usage, dummy_array))
predicted_cpu_usage = scaler.inverse_transform(full_input)[:, 0]
actual_cpu_usage = actual_cpu_usage.reshape(-1, 1)
full_input_actual = np.hstack((actual_cpu_usage, dummy_array))  # Combine with dummy array
actual_cpu_usage = scaler.inverse_transform(full_input_actual)[:, 0]  # Get the inverse transformed CPU usage
print('predicted_cpu_usage: ', predicted_cpu_usage[:10])
print('actual_cpu_usage: ', actual_cpu_usage[:10])

print(f'Test MSE: {test_loss.item()}')
print(f'Test R-squared: {r2}')

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(actual_cpu_usage, label='Actual CPU Usage', marker='o')
plt.plot(predicted_cpu_usage, label='Predicted CPU Usage', linestyle='--', marker='x')

plt.title('Comparison of Actual and Predicted CPU Usage')
plt.xlabel('Test Samples')
plt.ylabel('CPU Usage')
plt.legend()
plt.show()


