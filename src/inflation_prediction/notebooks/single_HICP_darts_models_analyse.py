# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd

# # Load the data

df = pd.read_csv('../data/filtered_data.csv')
print(f'Shape of the data: {df.shape}')
df.head()

# # Predictions done on a single HICP column

df_HICP = df.loc[:, ['month', 'HICP']]

print(f'Shape of X HICP: {df_HICP.shape}')
df_HICP.head()

# +
# convert df to dart TimeSeries
from darts import TimeSeries

TRAIN_MONTHS = df_HICP.shape[0]

series = TimeSeries.from_dataframe(
    df_HICP, time_col='month', value_cols='HICP'
)
# -

series.plot()

# # Split the data into train and test sets

# +
import matplotlib.pyplot as plt

train, val = series.split_after(TRAIN_MONTHS - 5)
plt.figure(figsize=(8, 5))
train.plot(label='train', marker='o')
val.plot(label='val', marker='o')
plt.title('Train and Validation Split')
plt.legend()


# -

def plot_forecast(series, forecast, title):
    plt.figure(figsize=(18, 8))
    series.plot(label='actual', lw=2, marker='o')
    forecast.plot(label='forecast', lw=2, marker='o')
    plt.title(title)
    plt.legend()
    plt.savefig(f'../images/{title}.png')
    plt.show()


# ## NaiveSeasonal

# +
from darts.models import NaiveSeasonal
from darts.metrics import mse

mse_metrics = []

# Initialize model
model = NaiveSeasonal(K=1)  # 1 month seasonality

# Fit model
model.fit(train)

# Predict
prediction = model.predict(n=4)
mse_naive_seasonal = mse(val, prediction)
mse_metrics.append(mse_naive_seasonal)

print(f'Prediction values scaled: {prediction.values()}')
print(f'MSE: {mse_naive_seasonal}')
# -

plot_forecast(train, prediction, 'Naive Seasonal Forecast compared to train')
plot_forecast(val, prediction, 'Naive Seasonal Forecast compared to actual')

# ## ARIMA

# +
from darts.models import ARIMA

# Initialize a model
model = ARIMA(p=1, d=1, q=1)

# Fit the model
model.fit(train)

# Predict
prediction = model.predict(n=4)
mse_arima = mse(val, prediction)
mse_metrics.append(mse_arima)

print(f'Prediction values: {prediction.values()}')
print(f'MSE: {mse_arima}')
# -

plot_forecast(train, prediction, 'ARIMA compared to train')
plot_forecast(val, prediction, 'ARIMA compared to actual')

# ## XGBModel

# +
from darts.models.forecasting.xgboost import XGBModel
from darts.metrics import mse

# Initialize a model
model = XGBModel(lags=3)

# Fit the model
model.fit(train)

# Predict
prediction = model.predict(n=4)

# evaluate the model
mse_xgb = mse(val, prediction)
mse_metrics.append(mse_xgb)

print(f'Prediction values: {prediction.values()}')
print(f'MSE: {mse_xgb}')
# -

plot_forecast(train, prediction, 'XGBoost compared to train')
plot_forecast(val, prediction, 'XGBoost compared to actual')

# ## Prophet

# +
from darts.models import Prophet

# Initialize a model
model = Prophet()

# Fit the model
model.fit(train)

# Predict
prediction = model.predict(n=4)

# evaluate the model
mse_prophet = mse(val, prediction)
mse_metrics.append(mse_prophet)

print(f'Prediction values: {prediction.values()}')
print(f'MSE: {mse_prophet}')
# -

plot_forecast(train, prediction, 'Prophet compared to train')
plot_forecast(val, prediction, 'Prophet compared to actual')
