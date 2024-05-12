#%% md
# # Load the data
#%%
import pandas as pd
#%%
df = pd.read_csv('../data/filtered_data.csv', usecols=lambda x: x != 'dump')
print(f'Shape of the data: {df.shape}')
df.head()
#%%
# selected feature names from feature selection
feature_names = [
    'month',
    'HICP',
    'Bieżący wskaźnik ufności konsumenckiej (BWUK)',
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
    'Produkcja sprzedana przemysłu ogółem (ceny stałe, B)'
]
#%%
df = df.loc[:, feature_names]
print(f'Shape of the data: {df.shape}')
df.head()
#%%
y = df.loc[:, 'HICP']
X = df.drop(columns=['month', 'HICP'])
#%% md
# # Data scaling
#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=feature_names[2:])
X.head()
#%% md
# # Train test split
#%%
test_size = int(0.2 * df.shape[0])
full_time = df.loc[:, 'month']
test_time = df.loc[df.shape[0] - test_size:, 'month']
#%%
X_train, X_test = X.loc[:df.shape[0] - test_size - 1, :], X.loc[df.shape[0] - test_size:, :]
y_train, y_test = y.loc[:df.shape[0] - test_size - 1], y.loc[df.shape[0] - test_size:]
#%%
X_test
#%%
y_test
#%% md
# ## Linear Regression
#%%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#%%
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
print(f'Predicted values: {y_pred}')

mse_loss = mean_squared_error(y, y_pred)
print(f'Mean Squared Error on training: {mse_loss}')
#%%
import matplotlib.pyplot as plt


def display_result(total_time, true_data, predicted_data, title, mse_value):
    plt.figure(figsize=(15, 8))
    plt.plot(total_time, true_data, label='True data')
    plt.plot(total_time, predicted_data, label='Predicted data')
    plt.grid()
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('HICP')
    plt.xticks(rotation=45)
    plt.legend()
    plt.text(0.9, 0.89, f'MSE: {mse_value:.2f}', horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=12)
    plt.savefig(f'../images/{title}.png')
    plt.show()
#%%
display_result(full_time, y, y_pred, 'Linear Regression', mse_loss)
#%%
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f'Predicted values: {y_pred}')

mse_loss = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse_loss}')
#%%
display_result(test_time, y_test, y_pred, 'Linear Regression', mse_loss)
#%% md
# ## Ridge Regression (L2 Regularization)
#%%
from sklearn.linear_model import Ridge

model = Ridge(alpha=0.01)
model.fit(X, y)

y_pred = model.predict(X)
print(f'Predicted values: {y_pred}')

mse_loss = mean_squared_error(y, y_pred)
print(f'Mean Squared Error on training: {mse_loss}')
#%%
display_result(full_time, y, y_pred, 'Ridge Regression', mse_loss)
#%%
model = Ridge(alpha=0.01)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f'Predicted values: {y_pred}')

mse_loss = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse_loss}')
#%%
display_result(test_time, y_test, y_pred, 'Ridge Regression', mse_loss)
#%% md
# ## Polynomial Regression with degree 3
#%%
from sklearn.preprocessing import PolynomialFeatures

degree = 3
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)
X_test_poly = poly_features.transform(X)

model = LinearRegression()
model.fit(X_poly, y)

y_pred = model.predict(X_poly)
print(f'Predicted values: {y_pred}')

mse_loss = mean_squared_error(y, y_pred)
print(f'Mean Squared Error: {mse_loss}')
#%%
display_result(full_time, y, y_pred, f'Polynomial Regression (degree={degree})', mse_loss)
#%%
poly_features = PolynomialFeatures(degree=degree)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)
print(f'Predicted values: {y_pred}')

mse_loss = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse_loss}')
#%%
display_result(test_time, y_test, y_pred, f'Polynomial Regression (degree={degree})', mse_loss)