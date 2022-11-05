#%%
from asyncio.base_subprocess import BaseSubprocessTransport
from re import X
import pandas as pd 
import numpy as np 
import altair as alt 
alt.data_transformers.enable(max_rows = None)
import sklearn
import matplotlib.pyplot as plt
import scipy


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
from sklearn.metrics import plot_confusion_matrix as confusion_matrix

from sklearn.datasets import load_breast_cancer
from keras.datasets import cifar10
import plotly.express as px
from yellowbrick.cluster import KElbowVisualizer as elbow
from yellowbrick.cluster import SilhouetteVisualizer

from scipy.stats import kurtosis

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    scoring=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

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
breast['target'] = breast.target.astype(int).astype(str)

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
        print('{},{}'.format(i,j))
        test = cifar.loc[cifar.target.isin([i,j])]
        test_data = pca(n_components=3).fit_transform(test.drop(columns = 'target'))
        #test_data = pd.DataFrame(test_data,columns = ['pca1','pca2','pca3'])
        sc = silhouette_score(test_data,test.target)
        print('Score for ({},{}): {}'.format(i,j,sc))
# Targets 9 & 4 are the best for this purpose
#%%
cifar = cifar.loc[cifar.target.isin([9,4])]
cifar_data = cifar.drop(columns = 'target')

#%%
#########################################################################################
# Part 1
#########################################################################################
# Custering Algorithms
# KMeans
breast_kmean = kmeans()
breast_kmean_viz = elbow(breast_kmean,k=(1,10))
breast_kmean_viz.fit(breast_data)
breast_kmean_viz.show()
breast_kmean = kmeans(3)
breast_kmean_viz = SilhouetteVisualizer(breast_kmean)
breast_kmean_viz.fit(breast_data)
breast_kmean_viz.show()
print('#' *10)
cifar_kmean = kmeans()
cifar_kmean_viz = elbow(cifar_kmean,k=(1,10))
cifar_kmean_viz.fit(cifar_data)
cifar_kmean_viz.show()
cifar_kmean = kmeans(3)
cifar_kmean_viz = SilhouetteVisualizer(cifar_kmean)
cifar_kmean_viz.fit(cifar_data)
cifar_kmean_viz.show()

#%%
# GMM
# Code to allow us to build an elbow model for GMM
from sklearn.base import ClusterMixin
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbow
class GMClusters(GaussianMixture, ClusterMixin):
    """Subclass of GaussianMixture to make it a ClusterMixin."""

    def fit(self, X):
        super().fit(X)
        self.labels_ = self.predict(X)
        return self

    def get_params(self, **kwargs):
        output = super().get_params(**kwargs)
        output["n_clusters"] = output.get("n_components", None)
        return output

    def set_params(self, **kwargs):
        kwargs["n_components"] = kwargs.pop("n_clusters", None)
        return super().set_params(**kwargs)

breast_em_viz = KElbow(GMClusters(), k=(1,10), force_model=True)
breast_em_viz.fit(breast_data)
breast_em_viz.show()

cifar_em_viz = KElbow(GMClusters(),k=(1,10), force_model = True)
cifar_em_viz.fit(cifar_data)
cifar_em_viz.show()

##########################################################
# Part 2
##########################################################
# %%
# PCA

breast_comp = 5
breast_pca = pca(n_components = breast_comp)
breast_pca_values = breast_pca.fit_transform(breast_data)
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(breast_pca.explained_variance_ratio_ * 100)
}
total = 0
for i, var in enumerate(breast_pca.explained_variance_ratio_ * 100):
    total += var
fig = px.scatter_matrix(
    breast_pca_values,
    labels=labels,
    dimensions=range(breast_comp),
    color=breast.target,
    title = f'Total Variance Explained: {total:.1f}%'
)
fig.update_traces(diagonal_visible=False)
fig.show()

cifar_comp = 5
cifar_pca = pca(n_components = cifar_comp)
cifar_pca_values = cifar_pca.fit_transform(cifar_data)
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(cifar_pca.explained_variance_ratio_ * 100)
}
total = 0
for i, var in enumerate(cifar_pca.explained_variance_ratio_ * 100):
    total += var
fig = px.scatter_matrix(
    cifar_pca_values,
    labels=labels,
    dimensions=range(cifar_comp),
    color=cifar.target,
    title = f'Total Variance Explained: {total:.1f}%'
)
fig.update_traces(diagonal_visible=False)
fig.show()



# %%
# ICA
from scipy.stats import kurtosis
breast_comp = 5
breast_pca = ica(n_components = breast_comp)
breast_pca_values = breast_pca.fit_transform(breast_data)
kurt = []
total = 0
for i in range(len(breast_pca_values[0])):
    k = kurtosis(breast_pca_values[:,i])
    print(k)
    kurt.append(k)
    total += k
labels = {
    str(i): f"K={var:.1f}"
    for i, var in enumerate(kurt)
}
fig = px.scatter_matrix(
    breast_pca_values,
    labels=labels,
    dimensions=range(breast_comp),
    color=breast.target,
    title = f'Total Kurtosis: {total:.1f}'
)
fig.update_traces(diagonal_visible=False)
fig.show()



cifar_comp = 5
cifar_pca = ica(n_components = cifar_comp)
cifar_pca_values = cifar_pca.fit_transform(cifar_data)
kurt = []
total = 0
for i in range(len(cifar_pca_values[0])):
    k = kurtosis(cifar_pca_values[:,i])
    print(k)
    kurt.append(k)
    total += k
labels = {
    str(i): f"K={var:.1f}"
    for i, var in enumerate(kurt)
}
fig = px.scatter_matrix(
    cifar_pca_values,
    labels=labels,
    dimensions=range(cifar_comp),
    color=cifar.target,
    title = f'Total Kurtosis: {total:.1f}'
)
fig.update_traces(diagonal_visible=False)
fig.show()

#%%
# RPA
from scipy.stats import kurtosis
breast_comp = 5
breast_pca = rpa(n_components = breast_comp)
breast_pca_values = breast_pca.fit_transform(breast_data)
kurt = []
total = 0
for i in range(len(breast_pca_values[0])):
    k = kurtosis(breast_pca_values[:,i])
    print(k)
    kurt.append(k)
    total += k
labels = {
    str(i): f"K={var:.1f}"
    for i, var in enumerate(kurt)
}
fig = px.scatter_matrix(
    breast_pca_values,
    labels=labels,
    dimensions=range(breast_comp),
    color=breast.target,
    title = f'Total Kurtosis: {total:.1f}'
)
fig.update_traces(diagonal_visible=False)
fig.show()

cifar_comp = 5
cifar_pca = rpa(n_components = cifar_comp)
cifar_pca_values = cifar_pca.fit_transform(cifar_data)
kurt = []
total = 0
for i in range(len(cifar_pca_values[0])):
    k = kurtosis(cifar_pca_values[:,i])
    print(k)
    kurt.append(k)
    total += k
labels = {
    str(i): f"K={var:.1f}"
    for i, var in enumerate(kurt)
}
fig = px.scatter_matrix(
    cifar_pca_values,
    labels=labels,
    dimensions=range(cifar_comp),
    color=cifar.target,
    title = f'Total Kurtosis: {total:.1f}'
)
fig.update_traces(diagonal_visible=False)
fig.show()

#%%
# LDA
breast_pca = lda()
breast_pca_values = pd.Series(breast_pca.fit_transform(breast_data,y = breast.target)[:,0])
temp_data = pd.concat([breast_pca_values,breast.target],axis = 1)
temp_data.columns = ['LDA_Values','Target']
k = kurtosis(breast_pca_values)
breast_lda = alt.Chart(temp_data,title = f'LDA with kurtosis: {k:.1f}').mark_boxplot().encode(
    alt.X('Target'),
    alt.Y('LDA_Values'),
    alt.Color('Target')
)
breast_lda.save('breast_lda.png')

cifar_pca = lda()
cifar_pca_values = pd.Series(cifar_pca.fit_transform(cifar_data,cifar.target)[:,0])
k = kurtosis(cifar_pca_values)

temp_data = pd.concat([cifar_pca_values.reset_index(drop = True),cifar.target.reset_index(drop = True)],axis = 1)
temp_data.columns = ['LDA_Values','Target']
breast_lda = alt.Chart(temp_data,title = f'LDA with kurtosis: {k:.1f}').mark_boxplot().encode(
    alt.X('Target:O'),
    alt.Y('LDA_Values'),
    alt.Color('Target:O')
)
breast_lda.save('cifar_lda.png')
#%%
##############################################################
# Part 3
##############################################################
# Generate lists to iterate over
cluster_list = [kmeans(3,random_state=42),em(3,random_state=42)]
cluster_name_list = ['KMeans','EM Or GMM']
dimension_list = [lda(n_components=1),pca(5,random_state = 42),ica(5,random_state = 42),rpa(5,random_state = 42)]
dimension_name_list = ['LDA','PCA','ICA','RCA']
# Start with breast data
data = breast_data
data_name = 'Breast Cancer'
target = breast.target

for c in range(len(cluster_name_list)):
    cluster = cluster_list[c]
    cluster_name = cluster_name_list[c]
    for d in range(len(dimension_name_list)):
        dimension = dimension_list[d]
        dimension_name = dimension_name_list[d]
        print('{} {}'.format(cluster_name,dimension_name))
        if dimension_name == 'LDA':
            dimension.fit(data,target)
            values = dimension.transform(data)
            # Sort through to find the most kurtotic value
            best = 0
            besti = 99
            if cluster_name == 'KMeans':
                cluster.set_params(n_clusters = 2).fit(values)
            elif cluster_name == 'EM Or GMM':
                cluster.set_params(n_components = 2).fit(values)
            preds = pd.Series(cluster.predict(values))
            temp_data = pd.concat([target.astype(int),preds],axis = 1)
            temp_data.columns = ['target','preds']
            temp_data.loc[temp_data.preds == 0,'preds'] = 2
            temp_data.loc[temp_data.preds == 1,'preds'] = 0
            temp_data.loc[temp_data.preds == 2,'preds'] = 1
            k = kurtosis(values)
            a = accuracy(temp_data.target,temp_data.preds).round(2)
            p = precision(temp_data.target,temp_data.preds).round(2)
            
            chart = alt.Chart(temp_data,title = "Breast {}-{} with Kurtosis {}\nAccuracy:{} Precision:{}".format(cluster_name,dimension_name,k,a,p)).mark_point().encode(
                alt.X('jitter_true:Q',title = 'True Class'),
                alt.Y('jitter_pred:Q',title = 'Predicted Class'),
            ).transform_calculate(
                jitter_true = '(sqrt(-2*log(random()))*cos(2*PI*random())/10)+datum.target',
                jitter_pred = '(sqrt(-2*log(random()))*cos(2*PI*random())/10)+datum.preds'
            )
            chart.save('breast_{}_{}.png'.format(cluster_name,dimension_name))

        else:
            # Reduce the data
            reduced_data = dimension.fit_transform(data)
            # get the best 2 dimensions that are the most kurtotic
            fbest = 0
            fbesti = 99
            sbest = 0
            sbesti = 0
            for i in range(len(reduced_data[0])):
                k = kurtosis(reduced_data[:,i])
                if abs(k) > abs(fbest):
                    #print('in 1')
                    sbest = fbest
                    sbesti = fbesti
                    fbest = k
                    fbesti = i
                if abs(k) > abs(sbest) and abs(k) < abs(fbest):
                    #print('in 2')
                    sbest = k
                    sbesti = i
                
                #print('New K: {}  \nFirst: {}  Second: {}\n'.format(abs(k),fbest,sbest))
            # Train cluster model
            best_data = reduced_data[:,[fbesti,sbesti]]
            cluster.fit(best_data)

            # Plot the decision boundary. For that, we will assign a color to each
            x_min, x_max = best_data[:, 0].min() - np.std(best_data[:, 0]), best_data[:, 0].max() + np.std(best_data[:, 0])
            y_min, y_max = best_data[:, 1].min() - np.std(best_data[:, 1]), best_data[:, 1].max() + np.std(best_data[:, 1])
            h = (x_max - x_min)/10  # point in the mesh [x_min, x_max]x[y_min, y_max].
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

            # Obtain labels for each point in mesh. Use last trained model.
            Z = cluster.predict(np.c_[xx.ravel(), yy.ravel()])

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

            plt.plot(best_data[:, 0], best_data[:, 1], "k.", markersize=2)
            # Plot the centroids as a white X
            if cluster_name == 'KMeans':
                centroids = cluster.cluster_centers_
            elif cluster_name == 'EM Or GMM':
                centroids = np.empty(shape=(cluster.n_components, best_data.shape[1]))
                for i in range(cluster.n_components):
                    density = scipy.stats.multivariate_normal(cov=cluster.covariances_[i], mean=cluster.means_[i]).logpdf(best_data)
                    centroids[i, :] = best_data[np.argmax(density)]

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
                f"{cluster_name}-{dimension_name} With Kurtosis of {fbest:.3f} & {sbest:.3f} on {data_name}\nCentroids are marked with white cross"
            )
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.xticks(())
            plt.yticks(())
            plt.savefig(f"breast_{cluster_name}_{dimension_name}.png")
            plt.show()
                

#%%
# Start with CIFAR data
data = cifar_data
data_name = 'CIFAR'
target = cifar.target
cluster_list = [kmeans(3,random_state=42),em(3,random_state=42)]
cluster_name_list = ['KMeans','EM Or GMM']
dimension_list = [lda(n_components=1),pca(5,random_state = 42),ica(5,random_state = 42),rpa(5,random_state = 42)]
dimension_name_list = ['LDA','PCA','ICA','RCA']
#cluster_list = [em(2,random_state=42)]
#cluster_name_list = ['EM Or GMM']
#dimension_list = [pca(5)]
#dimension_name_list = ['PCA']
for c in range(len(cluster_name_list)):
    cluster = cluster_list[c]
    cluster_name = cluster_name_list[c]
    for d in range(len(dimension_name_list)):
        dimension = dimension_list[d]
        dimension_name = dimension_name_list[d]
        print('{} {}'.format(cluster_name,dimension_name))
        if dimension_name == 'LDA':
            dimension.fit(data,target)
            values = dimension.transform(data)
            # Sort through to find the most kurtotic value
            best = 0
            besti = 99
            if cluster_name == 'KMeans':
                cluster.set_params(n_clusters = 2).fit(values)
            elif cluster_name == 'EM Or GMM':
                cluster.set_params(n_components = 2).fit(values)
            preds = pd.Series(cluster.predict(values))
            temp_data = pd.concat([target.reset_index(drop = True).astype(int),preds],axis = 1)
            temp_data.columns = ['target','preds']
            temp_data.loc[temp_data.target == 9,'target'] = 1
            temp_data.loc[temp_data.target == 4,'target'] = 0
            k = kurtosis(values)
            a = accuracy(temp_data.target,temp_data.preds).round(2)
            p = precision(temp_data.target,temp_data.preds).round(2)
            
            chart = alt.Chart(temp_data,title = "Breast {}-{} with Kurtosis {}\nAccuracy:{} Precision:{}".format(cluster_name,dimension_name,k,a,p)).mark_point().encode(
                alt.X('jitter_true:Q',title = 'True Class'),
                alt.Y('jitter_pred:Q',title = 'Predicted Class'),
            ).transform_calculate(
                jitter_true = '(sqrt(-2*log(random()))*cos(2*PI*random())/10)+datum.target',
                jitter_pred = '(sqrt(-2*log(random()))*cos(2*PI*random())/10)+datum.preds'
            )
            chart.save('cifar_{}_{}.png'.format(cluster_name,dimension_name))

        else:
            # Reduce the data
            reduced_data = dimension.fit_transform(data)
            # get the best 2 dimensions that are the most kurtotic
            fbest = 0
            fbesti = 99
            sbest = 0
            sbesti = 0
            for i in range(len(reduced_data[0])):
                k = kurtosis(reduced_data[:,i])
                if abs(k) > abs(fbest):
                    #print('in 1')
                    sbest = fbest
                    sbesti = fbesti
                    fbest = k
                    fbesti = i
                if abs(k) > abs(sbest) and abs(k) < abs(fbest):
                    #print('in 2')
                    sbest = k
                    sbesti = i
                
                #print('New K: {}  \nFirst: {}  Second: {}\n'.format(abs(k),fbest,sbest))
            # Train cluster model
            best_data = reduced_data[:,[fbesti,sbesti]]
            cluster.fit(best_data)

            # Plot the decision boundary. For that, we will assign a color to each
            x_min, x_max = best_data[:, 0].min() - np.std(best_data[:, 0]), best_data[:, 0].max() + np.std(best_data[:, 0])
            y_min, y_max = best_data[:, 1].min() - np.std(best_data[:, 1]), best_data[:, 1].max() + np.std(best_data[:, 0])
            h = (x_max-x_min)/10  # point in the mesh [x_min, x_max]x[y_min, y_max].
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

            # Obtain labels for each point in mesh. Use last trained model.
            Z = cluster.predict(np.c_[xx.ravel(), yy.ravel()])

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

            plt.plot(best_data[:, 0], best_data[:, 1], "k.", markersize=2)
            # Plot the centroids as a white X
            if cluster_name == 'KMeans':
                centroids = cluster.cluster_centers_
            elif cluster_name == 'EM Or GMM':
                centroids = np.empty(shape=(cluster.n_components, best_data.shape[1]))
                for i in range(cluster.n_components):
                    density = scipy.stats.multivariate_normal(cov=cluster.covariances_[i], mean=cluster.means_[i]).logpdf(best_data)
                    centroids[i, :] = best_data[np.argmax(density)]

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
                f"{cluster_name}-{dimension_name} With Kurtosis of {fbest:.3f} & {sbest:.3f} on {data_name}\nCentroids are marked with white cross"
            )
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.xticks(())
            plt.yticks(())
            plt.savefig(f"cifar_{cluster_name}_{dimension_name}.png")
            plt.show()
#%%
#%%
data = breast_data
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
####################################
# Determine best reduction
####################################
breast_data = breast.drop(columns = 'target')
# create chart that shows cluster distribution for the target data using LDA
breast_pca = lda()
breast_pca_values = pd.Series(breast_pca.fit_transform(breast_data,y = breast.target)[:,0])
temp_data = pd.concat([breast_pca_values,breast.target.reset_index(drop = True)],axis = 1)
temp_data.columns = ['LDA_Values','Target']
k = kurtosis(breast_pca_values)
breast_lda = alt.Chart(temp_data,title = f'Total Kurtosis For LDA: {k:.1f}').mark_boxplot().encode(
    alt.X('Target'),
    alt.Y('LDA_Values'),
    alt.Color('Target:O')
)
breast_lda.save('NNbreast_lda.png')
breast_lda

dimension_list = [pca(5),ica(5),rpa(5)]
dimension_name_list = ['PCA','ICA','RCA']
for j in range(len(dimension_name_list)):
    from scipy.stats import kurtosis
    breast_comp = 5
    breast_pca = dimension_list[j]
    breast_pca_values = breast_pca.fit_transform(breast_data)
    kurt = []
    sil = silhouette_score(breast_pca_values,breast.target)
    total = 0
    for i in range(len(breast_pca_values[0])):
        k = kurtosis(breast_pca_values[:,i])
        kurt.append(k)
        total += k
    labels = {
        str(i): f"K={var:.1f}"
        for i, var in enumerate(kurt)
    }
    fig = px.scatter_matrix(
        breast_pca_values,
        labels=labels,
        dimensions=range(breast_comp),
        color=breast.target,
        title = f'{dimension_name_list[j]} Kurtosis:{total:.1f}  Silhouette:{sil:.1f}'
    )
    fig.update_traces(diagonal_visible=False)
    #fig.write_image(f"target_{dimension_name_list[j]}.png")
    fig.show()

#%%

activationdf = []
solverdf = []
layersdf = []
itterdf = []
accuracyl = []
precisionl = []
layers = [2,4,6,8,10]
solver = ['sgd','adam']
activation = ['identity','logistic','relu']
itter = [200,250,300,350,400,450,500,550,600,650,700]
count = 1

x_train,x_test,y_train,y_test = train_test_split(breast.drop(columns = 'target'),breast.target,test_size = .2)

for l in layers:
    for s in solver:
        for a in activation:
            for i in itter:
                print('{} out of {}'.format(count,len(layers)*len(solver)*len(activation)*len(itter))) 
                model_mlp = mlp(hidden_layer_sizes=l,
                                activation= a,
                                solver= s,
                                max_iter=i)

                model_mlp.fit(x_train,y_train)


                pred = model_mlp.predict(x_test).astype(int)

                activationdf.append(a)
                solverdf.append(s)
                layersdf.append(l)
                itterdf.append(i)
                accuracyl.append(accuracy(y_test.astype(int),pred))
                precisionl.append(precision(y_test.astype(int),pred))
                print('\n\n')

                count = count + 1
results_mlp = pd.DataFrame({'activation':activationdf,
                           'solver':solverdf,
                           'layers':layersdf,
                           'iterations':itterdf,
                           'accuracy':accuracyl,
                           'precision':precisionl})
#%%
ann = mlp(activation = 'logistic',max_iter = 900,solver = 'adam',hidden_layer_sizes = 2)
plot_learning_curve(ann,
                    title = 'Base Estimator',
                    X = breast.drop(columns = 'target'),
                    y = breast.target)
# %%
dimension_list = [pca(5),ica(5),rpa(5),lda(n_components = 1)]
dimension_name_list = ['PCA','ICA','RCA','LDA']
for i in range(len(dimension_name_list)):
    dim = dimension_list[i]
    if dimension_name_list[i] == 'LDA':
        dim.fit(breast_data,breast.target)
    else:
        dim.fit(breast_data)
    new_val = dim.transform(breast_data)

    data = pd.DataFrame(new_val)
    fkmeans = pd.Series(kmeans(n_clusters=3).fit_predict(data))
    #optional here to add in one lda column
    data_final = pd.concat([data,fkmeans,breast.reset_index(drop = True).target],axis = 1)
    if dimension_name_list[i] == 'LDA':
        data_final.columns = ['comp1','kmeans_pred','target']
    else:
        data_final.columns = ['comp1','comp2','comp3','comp4','comp5','kmeans_pred','target']

    ann = mlp(activation = 'logistic',max_iter = 900,solver = 'adam',hidden_layer_sizes = 2)

    plot_learning_curve(ann,
                        title = '{} Reduction W/O KM Cluster Help'.format(dimension_name_list[i]),
                        X = data_final.drop(columns = ['target','kmeans_pred']),
                        y = data_final.target)

    plot_learning_curve(ann,
                        title = '{} Reduction W KM Cluster Help'.format(dimension_name_list[i]),
                        X = data_final.drop(columns = 'target'),
                        y = data_final.target)   
#%%
dimension_list = [pca(5),ica(5),rpa(5),lda(n_components = 1)]
dimension_name_list = ['PCA','ICA','RCA','LDA']
for i in range(len(dimension_name_list)):
    dim = dimension_list[i]
    if dimension_name_list[i] == 'LDA':
        dim.fit(breast_data,breast.target)
    else:
        dim.fit(breast_data)
    new_val = dim.transform(breast_data)

    data = pd.DataFrame(new_val)
    fkmeans = pd.Series(em(n_components=3).fit_predict(data))
    #optional here to add in one lda column
    data_final = pd.concat([data,fkmeans,breast.reset_index(drop = True).target],axis = 1)
    if dimension_name_list[i] == 'LDA':
        data_final.columns = ['comp1','kmeans_pred','target']
    else:
        data_final.columns = ['comp1','comp2','comp3','comp4','comp5','kmeans_pred','target']

    ann = mlp(activation = 'logistic',max_iter = 900,solver = 'adam',hidden_layer_sizes = 2)

    plot_learning_curve(ann,
                        title = '{} Reduction W/O EM Cluster Help'.format(dimension_name_list[i]),
                        X = data_final.drop(columns = ['target','kmeans_pred']),
                        y = data_final.target)

    plot_learning_curve(ann,
                        title = '{} Reduction W EM Cluster Help'.format(dimension_name_list[i]),
                        X = data_final.drop(columns = 'target'),
                        y = data_final.target) 
# %%
