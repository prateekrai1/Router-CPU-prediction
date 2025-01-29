import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam

train_df = pd.read_csv(r"E:\Router CPU prediction\data\train.csv", parse_dates=['timestamp'], index_col='timestamp')
test_df = pd.read_csv(r"E:\Router CPU prediction\data\test.csv", parse_dates=['timestamp'], index_col='timestamp')

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

def create_sequences(data, time_steps=5):
    x,y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(x), np.array(y)

time_steps = 5

x_train, y_train = create_sequences(train_scaled, time_steps)
x_test, y_test = create_sequences(test_scaled, time_steps)

model = Sequential([
    SimpleRNN(units=50, activation="tanh", return_sequences=True, input_shape=(time_steps, 1)),
    Dropout(0.2),
    SimpleRNN(units=50, activation='tanh'),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'mean_squared_errror')
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

y_pred = model.predict(x_test)

y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_actual = scaler.inverse_transform(y_pred)

plt.figure(figsize=(10,5))
plt.plot(y_test_actual, label='Actual CPU Usage')
plt.plot(y_pred_actual, label='Predicted CPU Usage', linestyle='dashed')
plt.legend()
plt.title("CPU Usage Prediction")
plt.Xlabel("Time Steps")
plt.ylabel("CPU Usage")

plt.show()