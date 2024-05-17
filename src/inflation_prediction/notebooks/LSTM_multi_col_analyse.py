#%% md
# # Load the data
#%%
import pandas as pd
#%%
df = pd.read_csv("filtered_data.csv")
print(f"Shape of the data: {df.shape}")
df.head()
#%%
selected_features = ['Bieżący wskaźnik ufności konsumenckiej (BWUK)',
                     'Import towarów (ceny bieżące w mln zł)',
                     'Koniunktura - przetwórstwo przemysłowe',
                     'Przeciętne zatrudnienie w sektorze przedsiębiorstwa (w w tys.)',
                     'Wydatki budżetu państwa (w mln zł)',
                     'Produkcja - dobra konsumpcyjne nietrwałe (B)',
                     'Przeciętne miesięczne nominalne wynagrodzenie brutto w sektorze przedsiębiorstwa (w zł)',
                     'Stopa bezrobocia rejestrowanego (%)',
                     'Przeciętna miesięczna nominalna emerytura i renta brutto z pozarolniczego systemu ubezpieczeń społecznych (w zł)',
                     'Koniunktura - działalność finansowa i ubezpieczeniowa',
                     'Wskaźniki cen skupu mleka (B)',
                     'Produkcja sprzedana przemysłu ogółem (ceny stałe, B)']
#%%
df = df.loc[:, ['month', 'HICP'] + selected_features]
df.head()
#%%
df['month'] = pd.to_datetime(df['month'])
df.set_index('month', inplace=True)
print(f"Shape of the data: {df.shape}")
df.head()
#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df.loc[:, 'HICP'])
plt.title("HICP over time")
plt.xlabel("Time")
plt.ylabel("HICP")
plt.show()
#%%
df.info()
#%% md
# # Train-Test Split
#%%
test_split = round(len(df) * 0.2)

df_for_training = df[:-test_split]
df_for_testing = df[-test_split:]
df_6_months_future = df[-test_split + 6:]
print(f'Shape of train_data: {df_for_training.shape}')
print(f'Shape of test_data: {df_for_testing.shape}')
print(f'Shape of future_prediction_data: {df_6_months_future.shape}')
#%%
df_for_training.tail()
#%%
df_for_testing.head()
#%% md
# # Scale the data
#%%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled = scaler.transform(df_for_testing)
print(f'Shape of train_data: {df_for_training_scaled.shape}')
print(f'Shape of test_data: {df_for_testing_scaled.shape}')
#%% md
# # Create Sequences
#%%
import numpy as np


def create_sequences(dataset, seq_length, df_indices):
    X, y = [], []
    indices = []
    for i in range(seq_length, len(dataset)):
        X.append(dataset[i - seq_length:i, 0:dataset.shape[1]])
        y.append(dataset[i, 0])
        indices.append(df_indices[i])

    return np.array(X), np.array(y), np.array(indices)
#%%
seq_length = 2
X_train, y_train, train_indices = create_sequences(df_for_training_scaled, seq_length, df_for_training.index)
X_test, y_test, test_indices = create_sequences(df_for_testing_scaled, seq_length, df_for_testing.index)
#%%
print(f'Shape of the training data: {X_train.shape}, {y_train.shape}, {train_indices.shape}')
print(f'Shape of the testing data: {X_test.shape}, {y_test.shape}, {test_indices.shape}')
#%% md
# # LSTM model architecture
#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
#%%
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
#%%
import tensorflow as tf


def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


# compile the model with adam optimizer and mean squared error loss
model.compile(optimizer='adam', loss=root_mean_squared_error)
#%%
model.summary()
#%%
# fit the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=1)
#%%
last_loss_value = history.history['loss'][-1]
print(f'Last loss value: {last_loss_value}')
#%%
train_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)
#%%
print(f'Shape of train_prediction: {train_prediction.shape}')
print(f'Shape of test_prediction: {test_prediction.shape}')
#%%
# inverse transform the scaled data, repeat the prediction to match the shape of the original data
train_prediction_copies = np.repeat(train_prediction, df_for_training.shape[1], axis=-1)
train_prediction = scaler.inverse_transform(
    np.reshape(train_prediction_copies, (len(train_prediction), df_for_training.shape[1])))[:, 0]
print(f'Shape of the train prediction: {train_prediction.shape}')
#%%
test_prediction_copies = np.repeat(test_prediction, df_for_testing.shape[1], axis=-1)
test_prediction = scaler.inverse_transform(
    np.reshape(test_prediction_copies, (len(test_prediction), df_for_testing.shape[1])))[:, 0]
print(f'Shape of the test prediction: {test_prediction.shape}')
#%%
original_copies_train = np.repeat(y_train, df_for_training.shape[1], axis=-1)
original_train = scaler.inverse_transform(np.reshape(original_copies_train, (len(y_train), df_for_training.shape[1])))[
                 :, 0]
print(f'Shape of the original: {original_train.shape}')
#%%
original_copies_test = np.repeat(y_test, df_for_testing.shape[1], axis=-1)
original_test = scaler.inverse_transform(np.reshape(original_copies_test, (len(y_test), df_for_testing.shape[1])))[:, 0]
print(f'Shape of the original: {original_test.shape}')
#%%
plt.figure(figsize=(22, 12))

plt.plot(train_indices, original_train, label='Actual HICP', color='blue')
plt.plot(train_indices, train_prediction, label='Test Prediction', color='red')
plt.text(0.05, 0.92, f'RMSE: {last_loss_value:.4f}', horizontalalignment='center', verticalalignment='center',
         transform=plt.gca().transAxes, fontsize=12)

plt.title('HICP Inflation Prediction on Train data')
plt.xlabel('Time')
plt.ylabel('HICP')
plt.legend()
plt.show()
#%%
plt.figure(figsize=(22, 12))

plt.plot(test_indices, original_test, label='Actual HICP', color='blue')
plt.plot(test_indices, test_prediction, label='Test Predictions', color='red')

plt.title('HICP Inflation Prediction on Test Data')
plt.xlabel('Time')
plt.ylabel('HICP')
plt.legend()
plt.show()
#%% md
# The blue line represents the actual HICP number.
# The orange line represents the forecasted HICP.
# Potential Overfitting: It is difficult to assess the accuracy of the forecast from this visualization alone.
#%% md
# # Predicting future values
#%%
future_prediction_split = 6
df_6_months_past = df.iloc[-2 * future_prediction_split:-future_prediction_split, :]
print(f'Shape of the 6 months past data: {df_6_months_past.shape}')
df_6_months_past.tail()
#%%
df_6_months_future_gap = df_6_months_future['HICP'].values.copy()
df_6_months_future_gap
#%%
df_6_months_future.loc[:, 'HICP'] = 0
df_6_months_future.tail()
#%%
old_scaled_array = scaler.transform(df_6_months_past)
new_scaled_array = scaler.transform(df_6_months_future)
#%%
new_scaled_df = pd.DataFrame(new_scaled_array)
new_scaled_df.iloc[:, 0] = np.nan
full_df = pd.concat([pd.DataFrame(old_scaled_array), new_scaled_df]).reset_index().drop(["index"], axis=1)
print(f'Shape of the full data: {full_df.shape}')
#%%
full_df.tail()
#%%
full_df_scaled_array = full_df.values
all_data = []
time_step = 6

for i in range(time_step, len(full_df_scaled_array)):
    data_x = []
    data_x.append(full_df_scaled_array[i - time_step:i, 0:full_df_scaled_array.shape[1]])
    data_x = np.array(data_x)
    prediction = model.predict(data_x)
    all_data.append(prediction)
    full_df.iloc[i, 0] = prediction
#%%
new_array = np.array(all_data)
new_array = new_array.reshape(-1, 1)
prediction_copies_array = np.repeat(new_array, df_for_testing.shape[1], axis=-1)
y_pred_future_6_months = scaler.inverse_transform(
    np.reshape(prediction_copies_array, (len(new_array), df_for_testing.shape[1])))[:, 0]
print(f'Shape of the future prediction: {y_pred_future_6_months.shape}')
#%%
plt.figure(figsize=(20, 10))
plt.plot(df_6_months_future.index, y_pred_future_6_months, color='blue', label='Predicted Global Active Power')
plt.plot(df_6_months_future.index, df_6_months_future_gap, color='red', label='Real Global Active Power')
plt.title('HICP Forecast for the next 6 months')
plt.xlabel('Time')
plt.ylabel('HICP')
plt.legend()
plt.show()