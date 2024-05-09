#%% md
# # Feature selection
#%% md
# ## Load the data
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
df = pd.read_csv('../data/filtered_data.csv', usecols=lambda x: x != 'dump')
print(f'Shape of the data: {df.shape}')
df.head()
#%%
df.info()
#%%
# pick only numerical columns
df = df.select_dtypes(include=['float64', 'int64'])
df.head()
#%% md
# ## Data normalization
#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
print(f'Shape of the scaled data: {X_scaled.shape}')
X_scaled[:3]
#%%
target_scaled = X_scaled[:, 0]
X_data = X_scaled[:, 1:]
#%% md
# ## PCA
#%%
from sklearn.decomposition import PCA
#%%
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_data)
pca.explained_variance_ratio_
#%%
plt.scatter(np.arange(5), pca.explained_variance_ratio_, c='r')
plt.show()
#%%
cols = df.columns[1:]
most_important_indices = [np.abs(pca.components_[i]).argmax() for i in range(pca.n_components_)]
pca_most_important_names = [cols[i] for i in most_important_indices]
pca_most_important_names
#%% md
# # RFE (Recursive Feature Elimination)
#%%
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
#%%
model = LinearRegression()
rfe = RFE(model, n_features_to_select=5)
fit = rfe.fit(X_data, target_scaled)
selected_features = fit.ranking_

rfe_most_important_names = cols[selected_features == 1].tolist()
rfe_most_important_names
#%% md
# ## Feature importance
#%%
from sklearn.ensemble import RandomForestRegressor
#%%
model = RandomForestRegressor(n_estimators=100)

model.fit(X_data, np.ravel(target_scaled))

feature_importance = model.feature_importances_
most_important_indices = np.argsort(feature_importance)[::-1][:5]
feat_im_most_important_nams = [cols[i] for i in most_important_indices]
feat_im_most_important_nams
#%% md
# ## Select the most important features from all of the methods
#%%
most_important_names = pca_most_important_names + rfe_most_important_names + feat_im_most_important_nams
most_important_features = list(set(most_important_names))

print(f'Length of the most important features: {len(most_important_features)}')
most_important_features