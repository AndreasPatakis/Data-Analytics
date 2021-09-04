"""TO-DO/TO-KNOW:
    SCALERS:
        MINMAX,
        Standard Scaler
    CLUSTERS:
        KMeans,
        DBSCAN,
        OPTICS,
        Birch
    METRICS:
        Confusion Matrix,
        Fowlkes Mallows,
        Normalized Mutual Info,
        Completeness
"""

"""APPROACH: Use wide range of values for classification(the k) for all our clustering methods and score them
using every metrics that we include. The overall-best K-value wins. Also do the above for different number of
features reduced by PCA."""

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,fowlkes_mallows_score,completeness_score
import matplotlib as plt
import pandas as pd
import numpy as np

def subtract_one(df,feature):
    """Subtracting 1 from a column in a dataframe"""

    df[feature] = df[feature].sub(1)
    return df

def evaluate(observed,predicted):
    """Evaluation of the classification using a number of different metrics"""

    conf_matrix_res = confusion_matrix(observed,predicted)
    print('Confusion Matrix results:\n{}\n'.format(conf_matrix_res))
    fowlkes_mallows_eval = fowlkes_mallows_score(observed,predicted)
    print('Fowlkes Mallows evaluation score:\n{}\n'.format(fowlkes_mallows_eval))
    completeness_eval = completeness_score(observed,predicted)
    print('Completeness evaluation score:\n{}\n'.format(completeness_eval))



def kmeans_clustering(df,features,k):
    """Applies KMeans algorithm for the given K"""

    #CLASS column scales from [1-10], Kmeans.labels are in [0-9] form so we subtract CLASS by 1
    df = subtract_one(df,'CLASS')

    kmeans = KMeans(n_clusters=k,init='k-means++',n_init=10,max_iter=500,tol=0.0001)
    kmeans = kmeans.fit(df[features])
    evaluate(observed=df['CLASS'], predicted=kmeans.labels_)


if __name__ == '__main__':

    """Importing the dataset"""
    df = pd.read_csv('./Data/cleanedCTG.csv')
    features = ['LB','AC','FM','UC','DL','DS','DP','ASTV','MSTV','ALTV','MLTV','Width','Min','Max','Nmax','Nzeros','Mode','Mean','Median','Variance','Tendency']
    features_with_classes = features + ['CLASS', 'NSP']

    """We will try 20 different k values for n_clusters and keep the best-scored one"""
    n_clusters = list(range(2,21))
    kmeans_clustering(df=df,features=features,k=5)
    #for k in n_clusters:
