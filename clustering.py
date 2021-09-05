"""TO-KNOW:
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

    TO-DO:
        CHECK AGAIN THE VALIDITY OF PCA_FEATURES.CSV
        BECAUSE MISSING VALUES WHERE FOUND AND THAT
        DOES NOT MAKE SENSE,
        MAKE THE 3D SCATTER PLOTS
"""

"""APPROACH: Use wide range of values for classification(the k) for all our clustering methods and score them
using every metrics that we include. The overall-best K-value wins. Also do the above for different number of
features reduced by PCA."""

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import confusion_matrix,fowlkes_mallows_score,completeness_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class cluster():
    """Template class for common uses of a cluster"""

    def __init__(self,cluster_algo):
        self.cluster_algo = cluster_algo
        self.conf_matrix = dict()
        self.results_df = pd.DataFrame()

    def evaluate_cluster(self,observed,predicted,k=0):
        """Evaluation of the classification using a number of different metrics"""

        res_dict = dict()
        if k:
            res_dict['K'] = k

        res_dict['Algorithm'] = self.cluster_algo

        conf_matrix_res = confusion_matrix(observed,predicted)
        self.save_conf_matrix(conf_matrix_res,k)

        fowlkes_mallows_eval = fowlkes_mallows_score(observed,predicted)
        res_dict['Fowlkes Mallows'] = fowlkes_mallows_eval

        completeness_eval = completeness_score(observed,predicted)
        res_dict['Completeness'] = completeness_eval

        eval_df = pd.DataFrame([res_dict])
        self.add_result(eval_df)

        if k:
            print('Evaluation completed for k={}.'.format(k))
        else:
            print("Evalutuation completed.")

        return eval_df

    def save_conf_matrix(self, conf_matrix_res,k):
        """Saves conf_matrix of the given instance"""

        self.conf_matrix[k] = conf_matrix_res

    def add_result(self, df):
        """Adds an evalaluation result to class's DataFrame for results"""

        self.results_df = self.results_df.append(df, ignore_index=True)

    def get_results(self, plot=False):
        """Returns and plots the results of the evaluation"""
        if plot:
            if 'K' in self.results_df.columns:
                figure, axis = plt.subplots(1,len(self.results_df.iloc[0,2:]))
                figure.suptitle("Algorithm applied: "+self.cluster_algo)
                figure.tight_layout()
                for i,col in enumerate(self.results_df.iloc[:,2:]):
                    x_values = self.results_df['K']
                    y_values = self.results_df[col]
                    axis[i].bar(x_values,y_values, align='center', alpha=0.7)
                    axis[i].set_xticks(x_values)
                    axis[i].set_title(col)
                    axis[i].set_xlabel('Value of K')
                    axis[i].set_ylabel('Score')
                print(self.results_df)
                plt.show()
            else:
                print('Nothing to plot.\nEvaluation results for {}:'.format(self.cluster_algo))
                print(self.results_df)

        return self.results_df

    def get_confusion_matrix(self,k=0):
        """Returns the confusion matrix of a given k"""
        return self.conf_matrix[k]

def subtract_one(df,feature):
    """Subtracting 1 from a column in a dataframe"""

    df[feature] = df[feature].sub(1)
    return df

def kmeans_clustering(df,features,k):
    """Applies KMeans algorithm for the given K"""

    kmeans = KMeans(n_clusters=k,init='k-means++',n_init=10,max_iter=500,tol=0.0001)
    kmeans = kmeans.fit(df[features])
    return kmeans

def agglomerative_clustering(df,features,k):
    """Applies the Hierachical Agglomerative algorithm for the given K"""

    hac = AgglomerativeClustering(n_clusters=k,linkage='single')
    hac = hac.fit(df[features])
    return hac

def DBSCAN_clustering(df,features):
    """Applies the DBSCAN algorithm to the given dataset"""


    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan = dbscan.fit(df[features])
    return dbscan


if __name__ == '__main__':

    # """Importing the dataset"""
    # df = pd.read_csv('./Data/cleanedCTG.csv')
    # features = ['LB','AC','FM','UC','DL','DS','DP','ASTV','MSTV','ALTV','MLTV','Width','Min','Max','Nmax','Nzeros','Mode','Mean','Median','Variance','Tendency']
    # features_with_classes = features + ['CLASS', 'NSP']
    #
    # """We will try 20 different k values for n_clusters and keep the best-scored one"""
    # n_clusters = list(range(2,21))
    #
    # kmeans_clusters = cluster(cluster_algo='kmeans')
    # hac_clusters = cluster(cluster_algo='Hierarchical Agglomerative')
    # dbscan_cluster = cluster(cluster_algo='DBSCAN')
    #
    # """CLASS,NSP columns scale from [1-10]. CLUSTERS.lables_ scale in [0-9] so we subtract CLASS or NSP by 1"""
    # y_label = 'CLASS'
    # df = subtract_one(df,y_label)
    # # for k in n_clusters:
    #
    #     # cluster = kmeans_clustering(df=df,features=features,k=k)
    #     # kmeans_clusters.evaluate_cluster(observed=df[y_label], predicted=cluster.labels_, k=k)
    #     #
    #     # cluster = agglomerative_clustering(df=df,features=features,k=k)
    #     # hac_clusters.evaluate_cluster(observed=df[y_label], predicted=cluster.labels_, k=k)
    #
    #
    #
    #
    # #kmeans_clusters.get_results(plot=True)
    # # hac_clusters.get_results(plot=True)
    # # print(hac_clusters.get_confusion_matrix(3))
    #
    # """Applying DBSCAN to our dataset"""
    # cluster = DBSCAN_clustering(df=df,features=features)
    # dbscan_cluster.evaluate_cluster(observed=df[y_label], predicted=cluster.labels_)
    # dbscan_cluster.get_results(plot=True)
    # print(dbscan_cluster.get_confusion_matrix())






    """Now we are applying the same algorithms but with the features scaled down to 3 using PCA"""






    df_pca = pd.read_csv('./Data/PCA_Features.csv')
    features = ['PCA1','PCA2','PCA3']
    features_with_classes = features + ['CLASS', 'NSP']


    """We will try 20 different k values for n_clusters and keep the best-scored one"""
    n_clusters = list(range(2,21))

    kmeans_clusters = cluster(cluster_algo='kmeans')
    hac_clusters = cluster(cluster_algo='Hierarchical Agglomerative')
    dbscan_cluster = cluster(cluster_algo='DBSCAN')

    """CLASS,NSP columns scale from [1-10]. CLUSTERS.lables_ scale in [0-9] so we subtract CLASS or NSP by 1"""
    y_label = 'CLASS'
    df_pca = subtract_one(df_pca,y_label)
    for k in n_clusters:

        cluster = kmeans_clustering(df=df_pca,features=features,k=k)
        kmeans_clusters.evaluate_cluster(observed=df_pca[y_label], predicted=cluster.labels_, k=k)

        # cluster = agglomerative_clustering(df=df_pca,features=features,k=k)
        # hac_clusters.evaluate_cluster(observed=df_pca[y_label], predicted=cluster.labels_, k=k)




    kmeans_clusters.get_results(plot=True)
    #hac_clusters.get_results(plot=True)


    """Applying DBSCAN to our dataset"""
    cluster = DBSCAN_clustering(df=df_pca,features=features)
    dbscan_cluster.evaluate_cluster(observed=df_pca[y_label], predicted=cluster.labels_)
    dbscan_cluster.get_results(plot=True)
    print(dbscan_cluster.get_confusion_matrix())
