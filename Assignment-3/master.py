#%%
from re import X
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as ss
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import precision_score as precision
from sklearn.metrics import silhouette_score 

from sklearn.datasets import load_breast_cancer
from keras.datasets import cifar10
import plotly.express as px
from yellowbrick.cluster import KElbowVisualizer as elbow
from yellowbrick.cluster import SilhouetteVisualizer

#%%
#Load new data
breast = load_breast_cancer()
breast_data = breast.data
breast_target = breast.target

labels = np.reshape(breast_target,(569,1))
final_brest_data = np.concatenate([breast_data,labels],axis = 1)
columns = breast.feature_names
columns = np.append(columns, 'target')
breast = pd.DataFrame(final_brest_data)
breast.columns = columns

breast_data = breast.drop(columns = 'target')
x = ss().fit_transform(breast_data)
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
breast_data = pd.DataFrame(x,columns=feat_cols)
################################

# %%
# Scale and normalize
(x_train, y_train), (x_test,y_test) = cifar10.load_data()

########################################
xs_train = x_train/255.0
x_train_flat = xs_train.reshape(-1,3072)
feat_cols = ['pixel'+str(i) for i in range(x_train_flat.shape[1])]
cifar = pd.DataFrame(x_train_flat,columns=feat_cols)
cifar['target'] = y_train # Num targets = 10
#%%
# see which combo of 2 pics gives the best for the silhouette score
for i in range(10):
    for j in range(10):
        if i == j:
            continue
        #print('{},{}'.format(i,j))
        test = cifar.loc[cifar.target.isin([i,j])]
        test_data = kmeans(n_clusters=2).fit_transform(test.drop(columns = 'target'))
        sc = silhouette_score(test_data,test.target)
        print('Score for ({},{}): {}'.format(i,j,sc))
# Targets 9 & 4 are the best for this purpose
#%%
cifar = cifar.loc[cifar.target.isin([9,4])]
cifar_data = cifar.drop(columns = 'target')

#%%
#########################################################################################
#########################################################################################
# Custering Algorithms
# KMeans
breast_kmean = kmeans()
breast_kmean_viz = elbow(breast_kmean,k=(1,10))
breast_kmean_viz.fit(breast_data)
breast_kmean_viz.show()

cifar_kmean = kmeans()
cifar_kmean_viz = elbow(cifar_kmean,k=(1,10))
cifar_kmean_viz.fit(cifar_data)


#%%
##################################################################################
##################################################################################


cifar_kmeans = kmeans(n_clusters = 10)
cifar_kmeans.fit(cifar_data)
reduced_data_cifar = cifar_kmeans.transform(cifar_data)

preds = cifar_kmeans.predict(cifar_data)
cifar['kmean_preds'] = preds
print('cifar silhouette: {}'.format(silhouette_score(reduced_data_cifar,cifar.target)))

###########################

cifar_kmean = kmeans(n_clusters = 2)
reduced_data = cifar_kmean.fit_transform(cifar_data)
cifar_kmean.fit(reduced_data)
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .1  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = cifar_kmean.predict(np.c_[xx.ravel(), yy.ravel()])

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
centroids = cifar_kmean.cluster_centers_
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
    "Image Clusters\n"
    "Centroids are marked with white cross"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

#print('Accuracy: {}   Precision: {}'.format(accuracy(cifar.target,cifar.kmean_preds).round(2),precision(cifar.target,cifar.kmean_preds).round(2)))
# %%
# EM or GMM
gm = em(n_components = 2,random_state = 42)
breast_gm = gm.fit_predict(breast_data)
gm = em(n_components = 2,random_state = 46)
cifar_gm = gm.fit_predict(cifar_data)

cacc = accuracy(breast.target,breast_gm).round(2)
cprec = precision(breast.target,breast_gm).round(2)
sacc = accuracy(cifar.target,cifar).round(2)
sprec = precision(cifar.target,cifar).round(2)

print('Breast Cancer\nAccuracy: {}   Precision: {}\n'.format(cacc,cprec))
print('Cifar\nAccuracy: {}   Precision: {}'.format(sacc,sprec))

# %%
# PCA
import plotly.express as px
churn_clean = pd.concat([pos,neg])
churn_data = churn_clean.drop(columns = ['churn','credit_card','active_member','gender','Germany','Spain','product_2','product_3','product_4'])

churn_comp = 5
churn_pca = pca(n_components = churn_comp)
#churn_data = ss().fit_transform(churn_data)
churn_pca_values = churn_pca.fit_transform(churn_data)
churn_pca_values = ss().fit_transform(churn_pca_values)
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(churn_pca.explained_variance_ratio_ * 100)
}
fig = px.scatter_matrix(
    churn_pca_values,
    labels=labels,
    dimensions=range(churn_comp),
    color=churn_clean.churn
)
fig.update_traces(diagonal_visible=False)
fig.show()
print('S Score: {}'.format(silhouette_score(churn_pca_values,churn_clean.churn)))
#%%
subscribed_comp = 2
subscribed_pca = pca(n_components = subscribed_comp)
subscribed_pca_values = subscribed_pca.fit_transform(breast_data)
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(subscribed_pca.explained_variance_ratio_ * 100)
}
fig = px.scatter_matrix(
    subscribed_pca_values,
    labels=labels,
    dimensions=range(subscribed_comp),
    color=breast.target
)
fig.update_traces(diagonal_visible=False)
fig.show()



# %%
# ICA
churn_comp = 5
churn_ica = ica(n_components = churn_comp)
churn_ica_values = churn_ica.fit_transform(churn_data)

fig = px.scatter_matrix(
    churn_ica_values,
    dimensions=range(churn_comp),
    color=churn_clean.churn
)
fig.update_traces(diagonal_visible=False)
fig.show()
# %%
data = churn_data
reduced_data = ica(n_components=2).fit_transform(data)
fkmeans = kmeans(init="k-means++", n_clusters=2, n_init=4)
fkmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .01  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - .01, reduced_data[:, 0].max() + .01
y_min, y_max = reduced_data[:, 1].min() - .01, reduced_data[:, 1].max() + .01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = fkmeans.predict(np.c_[xx.ravel(), yy.ravel()])

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
centroids = fkmeans.cluster_centers_
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
    "K-means clustering on the digits dataset (PCA-reduced data)\n"
    "Centroids are marked with white cross"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
# %%

############################################################################
# Part 4 & 5
############################################################################
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

Z = fkmeans.predict(reduced_data)
acc = accuracy(churn_clean.churn,Z)
prec = precision(churn_clean.churn,Z)
print('Accuracy: {}   Precision: {}'.format(acc,prec))
# %%
from sklearn.model_selection import train_test_split
MLP2 = mlp(activation = 'logistic',max_iter = 600,solver = 'sgd',hidden_layer_sizes = 2)
data = churn_clean[['churn','credit_card','active_member','gender','Germany','Spain','product_2','product_3','product_4']]
data.index = range(len(data))
reduced_data = pd.DataFrame(pca(n_components=5).fit_transform(data.drop(columns = 'churn')),columns = ['a1','a2','3','4','5'])
fkmeans = pd.Series(kmeans(init="k-means++", n_clusters=2, n_init=4).fit_predict(churn_data))
data_final = pd.concat([data,reduced_data,fkmeans],axis = 1)


x_train,x_test,y_train,y_test = train_test_split(data_final.drop(columns = 'churn'),data_final.churn,test_size = .2)

MLP2.fit(x_train,y_train)

preds = MLP2.predict(x_test)

acc = accuracy(y_test,preds)
prec = precision(y_test,preds)

print('Accuracy: {}   Precision: {}'.format(acc,prec))




# %%
