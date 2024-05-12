#%%
import pandas as pd
#%%
df = pd.read_csv("filtered_data.csv")
print(f"Shape of the data: {df.shape}")
df.head()
#%%
df_hicp = df.loc[:, ['month', 'HICP']]
df_hicp.head()
#%%
df_hicp['month'] = pd.to_datetime(df_hicp['month'])
df_hicp.set_index('month', inplace=True)
df_hicp.head()
#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df_hicp)
plt.title("HICP over time")
plt.xlabel("Time")
plt.ylabel("HICP")
#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_hicp)
#%%
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]
#%%
import numpy as np

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])

    return np.array(X), np.array(y)
#%%
seq_length = 5
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)
#%%
X_train.shape
#%% md
# We define an LSTM model using Sequential() and add an LSTM layer with 50 units, ReLU activation function, and input shape (seq_length, 1) where seq_length is the length of the input sequences with Dense layer with one unit (for regression) to output the predicted value compiling the model with Adam optimizer and mean squared error loss function.
#%%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
#%% md
# Architecture 1
#%%
model = Sequential([
    Input(shape=(seq_length, 1)),
    LSTM(units=50, activation='relu'),
    Dense(units=1)
])
#%% md
# Architecture 2
#%%
model = Sequential([
    Input(shape=(seq_length, 1)),
    LSTM(units=128, return_sequences=True, activation='relu'),
    LSTM(units=64, return_sequences=True, activation='relu'),
    LSTM(units=32, return_sequences=True, activation='relu'),
    Dense(units=1)
])
#%%
from keras import backend as K

def RMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

model.compile(optimizer='adam', loss=RMSE)
history = model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=1)
#%%
last_loss_value = history.history['loss'][-1]
print(f'Last loss value: {last_loss_value}')
#%%
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
#%%
train_predictions = scaler.inverse_transform(np.squeeze(train_predictions))
test_predictions = scaler.inverse_transform(np.squeeze(test_predictions))
#%%
train_predictions = np.mean(train_predictions, axis=1, keepdims=True)
test_predictions = np.mean(test_predictions, axis=1, keepdims=True)
#%%
plt.figure(figsize=(10, 6))

# Plot actual data
plt.plot(df_hicp.index[seq_length:], df_hicp['HICP'][seq_length:], label='Actual', color='blue')

# Plot training predictions
plt.plot(df_hicp.index[seq_length:seq_length+len(train_predictions)], train_predictions, label='Train Predictions',color='green')

test_pred_index = range(seq_length+len(train_predictions), seq_length+len(train_predictions)+len(test_predictions))
plt.plot(df_hicp.index[test_pred_index], test_predictions, label='Test Predictions',color='orange')
plt.text(0.1, 0.8, f'RMSE: {last_loss_value:.4f}', horizontalalignment='center', verticalalignment='center',
         transform=plt.gca().transAxes, fontsize=12)

plt.title('HICP Inflation Forecasting')
plt.xlabel('Time')
plt.ylabel('HICP')
plt.legend()
plt.savefig('LSTM.png')
plt.show()
#%% md
# The blue line represents the actual HICP number.
# The orange line represents the forecasted HICP.
# Potential Overfitting: It is difficult to assess the accuracy of the forecast from this visualization alone.
#%%
forecast_period = 5
forecast = []

# Use the last sequence from the test data to make predictions
last_sequence = X_test[-1]

for _ in range(forecast_period):
    # Reshape the sequence to match the input shape of the model
    current_sequence = last_sequence.reshape(1, seq_length, 1)
    # predict the next value
    next_prediction = model.predict(current_sequence)[0][0]
    # append the prediction to the forecast list
    forecast.append(next_prediction)
    # update the last sequence by removing the first element and adding the prediction
    last_sequence = np.append(last_sequence[1:], next_prediction)

# inverse transform
forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
#%%
forecast
#%%
plt.figure(figsize=(10, 6))
plt.plot(df_hicp.index[-len(test_data):], scaler.inverse_transform(test_data), label='Actual')
plt.plot(pd.date_range(start=df_hicp.index[-1], periods=forecast_period, freq='M'), forecast, label='Forecast')
plt.title('HICP Time Series Forecasting')
plt.xlabel('Year')
plt.ylabel('HICP')
plt.legend()
plt.savefig('LSTM_pred.png')
plt.show()