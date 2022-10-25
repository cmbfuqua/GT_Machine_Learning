#%%
import pandas as pd 
import numpy as np 
import altair as alt 
import sklearn
import matplotlib.pyplot as plt


from sklearn.neural_network import MLPClassifier as mlp
# clustering algorithms
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as em
# dimensionality reduction algorithms
from sklearn.decomposition import PCA as pca
from sklearn.decomposition import FastICA as ica
from sklearn.random_projection import GaussianRandomProjection as rpa
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import precision_score as precision
from sklearn.metrics import silhouette_score 

# %%
# Import our datasets
churn = pd.read_csv('churn.csv')
subscribed = pd.read_csv('subscribed.csv')
subscribed['subscribed'] = subscribed.term_deposit_subscribed
subscribed = subscribed.drop(columns = 'term_deposit_subscribed')
##################### clean the datasets
# Clean Subscriber
data_raw = pd.get_dummies(subscribed, columns = ['job_type','marital','education','default', 'prev_campaign_outcome'])
# Transformation of date time: https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
data_raw['sin_time'] = np.sin(2*np.pi*data_raw.day_of_month/365)
data_raw['cos_time'] = np.cos(2*np.pi*data_raw.day_of_month/365)
data_raw['housing_loan'] = np.where(data_raw['housing_loan'] == 'yes', 1,0)
data_raw['personal_loan'] = np.where(data_raw['personal_loan'] == 'yes', 1,0)
data_raw = data_raw.fillna(data_raw.mean())
data_cleaned = data_raw.drop(columns=['communication_type', 'day_of_month', 'month', 'id','days_since_prev_campaign_contact'])
# Downsample to balanced data
sub = data_cleaned.loc[data_cleaned.subscribed == 1]
nsub = data_cleaned.loc[data_cleaned.subscribed == 0].head(len(sub))
subscribed_clean = pd.concat([sub,nsub])

# Clean Churn
churn = churn.drop(columns = ['customer_id'])
churn['balancedSalary'] = churn.balance/churn.estimated_salary
churn['tenureAge'] = churn.tenure/churn.age
churn['creditAge'] = churn.credit_score/churn.age
churn.loc[churn.credit_card == 0,'credit_card'] = -1
churn.loc[churn.active_member == 0,'active_member'] = -1
clean = churn[['credit_score','age','tenure',
              'balance','credit_card','active_member',
              'estimated_salary','churn','balancedSalary',
              'tenureAge','creditAge']]
# one hot encode country
countries = pd.get_dummies(churn.country,drop_first=True)
# one hot encode gender
clean['gender'] = pd.get_dummies(churn.gender,drop_first=True)
# one hot encode the 4 products
products = pd.get_dummies(churn.products_number,drop_first=True,prefix='product')

clean = pd.concat([clean,countries,products],axis = 1)
#Downsample to balance the data
pos = clean.loc[clean.churn == 1]
neg = clean.loc[clean.churn == 0].head(len(pos))

churn_clean = pd.concat([pos,neg])

# %%
#########################################################################################
#########################################################################################
# Custering Algorithms
# KMeans
churn_clean = pd.concat([pos,neg])
churn_data = churn_clean.drop(columns = 'churn')

churn_kmean = kmeans(n_clusters = 2)
churn_kmean.fit(churn_data)

preds = churn_kmean.predict(churn_data)
churn_clean['predictions'] = preds

reduced_data = churn_kmean.fit_transform(churn_data)
churn_kmean.fit(reduced_data)
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 10  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
c_xx, c_yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = churn_kmean.predict(np.c_[c_xx.ravel(), c_yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(c_xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(c_xx.min(), c_xx.max(), c_yy.min(), c_yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
# Plot the centroids as a white X
centroids = churn_kmean.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)
plt.title(
    "Customer Churn Clusters\n"
    "Centroids are marked with white cross"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

##################################################################################
##################################################################################
subscribed_data = subscribed_clean.drop(columns = 'subscribed')

subscribed_kmean = kmeans(n_clusters = 2)
subscribed_kmean.fit(subscribed_data)

preds = subscribed_kmean.predict(subscribed_data)
subscribed_clean['predictions'] = preds

reduced_data = subscribed_kmean.fit_transform(subscribed_data)
subscribed_kmean.fit(reduced_data)
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 10  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = subscribed_kmean.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
# Plot the centroids as a white X
centroids = subscribed_kmean.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)
plt.title(
    "Subscribed Clusters\n"
    "Centroids are marked with white cross"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


#%%
print('Accuracy: {}   Precision: {}'.format(1-accuracy(churn_clean.churn,churn_clean.predictions).round(2),1-precision(churn_clean.churn,churn_clean.predictions).round(2)))
print('Accuracy: {}   Precision: {}'.format(accuracy(subscribed_clean.subscribed,subscribed_clean.predictions).round(2),precision(subscribed_clean.subscribed,subscribed_clean.predictions).round(2)))
print('churn silhouette: {}   subscribed silhouette: {}'.format(silhouette_score(churn_clean.drop(columns = 'churn'),churn_clean.churn),silhouette_score(subscribed_clean.drop(columns = 'subscribed'),subscribed_clean.subscribed)))
# %%
# EM or GMM
scaler = sklearn.preprocessing.StandardScaler()
scaled_churn = scaler.fit_transform(churn_data)
scaled_churn = pd.DataFrame(scaled_churn, columns = churn_data.columns)

scaled_subscribed = scaler.fit_transform(subscribed_data)
scaled_subscribed = pd.DataFrame(scaled_subscribed, columns = subscribed_data.columns)

gm = em(n_components = 2,random_state = 42)
churn_gm = gm.fit_predict(scaled_churn)
gm = em(n_components = 2,random_state = 46)
subscribed_gm = gm.fit_predict(scaled_subscribed)

cacc = accuracy(churn_clean.churn,churn_gm).round(2)
cprec = precision(churn_clean.churn,churn_gm).round(2)
sacc = accuracy(subscribed_clean.subscribed,subscribed_gm).round(2)
sprec = precision(subscribed_clean.subscribed,subscribed_gm).round(2)

print('Churn\nAccuracy: {}   Precision: {}\n'.format(cacc,cprec))
print('Subscribed\nAccuracy: {}   Precision: {}'.format(sacc,sprec))

# %%
# PCA
churn_pca = pca(n_components = 5)
churn_pca.fit(churn_data)
# %%
