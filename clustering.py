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
from mpl_toolkits.mplot3d import Axes3D
import random
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


class cluster():
    """Template class for common uses of a cluster"""

    def __init__(self,cluster_algo,dataframe):
        self.cluster_algo = cluster_algo
        self.df = dataframe
        self.conf_matrix = dict()
        self.results_df = pd.DataFrame()
        self.results_labels = dict()
        self.y_label = ''

    def evaluate_cluster(self,observed,y_label,predicted,k=0):
        """Evaluation of the classification using a number of different metrics"""

        self.y_label = y_label
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
        self.save_results_labels(predicted,k)

        if k:
            print('Evaluation of {} completed for k={}.'.format(self.cluster_algo,k))
        else:
            print("\nEvaluation of {} completed.".format(self.cluster_algo))

        return eval_df

    def save_conf_matrix(self, conf_matrix_res,k):
        """Saves conf_matrix of the given instance"""

        self.conf_matrix[k] = conf_matrix_res

    def save_results_labels(self,results,k):
        """saves result labels of the given cluster"""

        self.results_labels[k] = results

    def add_result(self, df):
        """Adds an evalaluation result to class's DataFrame for results"""

        self.results_df = self.results_df.append(df, ignore_index=True)

    def get_results(self, plot=False, save=False):
        """Returns, plots and saves the results of the evaluation"""

        if save:
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)

        if plot or save:
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
                if save:
                    plt.savefig(parent_dir+'/'+self.cluster_algo+'_'+self.y_label)
                    print("Hist was saved successfully.")
                if plot:
                    print('\n',self.results_df)
                    plt.show()
            else:
                print('\nNothing to or save plot.\nEvaluation results for {}:'.format(self.cluster_algo))
                print('\n',self.results_df)

        return self.results_df

    def get_confusion_matrix(self,k=0):
        """Returns the confusion matrix of a given k"""
        return self.conf_matrix[k]

    def get_results_labels(self,k=0):
        """Returns the labels predicted for a given k"""
        return self.results_labels[k]

    def plot_3dScatter(self,k=0,save=False,plot=False):
        if len(self.df.columns) != 3:
            return "Number of features must be 3.\n You gave: ",len(self.df.columns)
        if save:
            if not os.path.exists(parent_dir+'/3D_PLOTS'):
                os.makedirs(parent_dir+'/3D_PLOTS')

        def get_color():
            """Return a color in hex"""
            color = '#%06X' % random.randint(0, 0xFFFFFF)
            return color

        """Plot the datapoints of each cluster in 3d space"""

        labels = self.results_labels[k]
        distinct_labels = np.unique(labels)
        color_label = dict()
        for label in distinct_labels:
            color_label[label] = get_color()

        fig = plt.figure()
        ax = Axes3D(fig,auto_add_to_figure=False)
        fig.add_axes(ax)
        for l in distinct_labels:
            lx = np.where(l == labels)
            ax.scatter(self.df['PCA1'].loc[lx],self.df['PCA2'].loc[lx],self.df['PCA3'].loc[lx],label='Label: '+str(l+1), c=color_label[l])
        ax.legend()
        if save:
            plt.savefig(parent_dir+'/3D_PLOTS/3D_PLOT_'+self.cluster_algo+'_'+self.y_label+'_K_'+str(k))
            print("3D plot was saved successfully.")
        if plot:
            plt.show()

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

    parent_dir = './Clustering_Plots/All_Attributes'

    """Importing the dataset"""
    df = pd.read_csv('./Data/cleanedCTG.csv')
    features = ['LB','AC','FM','UC','DL','DS','DP','ASTV','MSTV','ALTV','MLTV','Width','Min','Max','Nmax','Nzeros','Mode','Mean','Median','Variance','Tendency']
    y_features = ['CLASS', 'NSP']
    features_with_classes = features + y_features

    """We will try 20 different k values for n_clusters and keep the best-scored one"""
    n_clusters = list(range(2,21))

    for y_label in y_features:

        print('\n',"-"*10,"Clustering for label: {} using all attributes".format(y_label),"-"*10,'\n')

        kmeans_clusters = cluster(cluster_algo='KMeans',dataframe=df[features])
        hac_clusters = cluster(cluster_algo='Hierarchical Agglomerative',dataframe=df[features])
        dbscan_cluster = cluster(cluster_algo='DBSCAN',dataframe=df[features])

        """CLASS,NSP columns scale from [1-10]. CLUSTERS.lables_ scale in [0-9] so we subtract CLASS or NSP by 1"""
        df = subtract_one(df,y_label)

        for k in n_clusters:

            cluster_obj = kmeans_clustering(df=df,features=features,k=k)
            kmeans_clusters.evaluate_cluster(observed=df[y_label],y_label=y_label, predicted=cluster_obj.labels_, k=k)
            kmeans_clusters.plot_3dScatter(k,save=True)

            cluster_obj = agglomerative_clustering(df=df,features=features,k=k)
            hac_clusters.evaluate_cluster(observed=df[y_label],y_label=y_label, predicted=cluster_obj.labels_, k=k)
            hac_clusters.plot_3dScatter(k,save=True)


        kmeans_clusters.get_results(plot=False,save=True)
        hac_clusters.get_results(plot=False, save=True)


        """Applying DBSCAN to our dataset"""
        cluster_obj = DBSCAN_clustering(df=df,features=features)
        dbscan_cluster.evaluate_cluster(observed=df[y_label], y_label=y_label,predicted=cluster_obj.labels_)
        dbscan_cluster.get_results(save=True)
        dbscan_cluster.plot_3dScatter(save=True)




    """Now we are applying the same algorithms but with the features scaled down to 3 using PCA"""



    parent_dir = './Clustering_Plots/PCA_Attributes'

    df_pca = pd.read_csv('./Data/PCA_Features.csv')
    features = ['PCA1','PCA2','PCA3']
    y_features = ['CLASS', 'NSP']
    features_with_classes = features + y_features


    """We will try 20 different k values for n_clusters and keep the best-scored one"""
    n_clusters = list(range(2,15))


    for y_label in y_features:
        print('\n',"-"*10,"Clustering for label: {} using PCA attributes".format(y_label),"-"*10,'\n')

        kmeans_clusters = cluster(cluster_algo='KMeans',dataframe=df_pca[features])
        hac_clusters = cluster(cluster_algo='Hierarchical Agglomerative',dataframe=df_pca[features])
        dbscan_cluster = cluster(cluster_algo='DBSCAN',dataframe=df_pca[features])

        """CLASS,NSP columns scale from [1-10]. CLUSTERS.lables_ scale in [0-9] so we subtract CLASS or NSP by 1"""
        df_pca = subtract_one(df_pca,y_label)
        for k in n_clusters:

            cluster_obj = kmeans_clustering(df=df_pca,features=features,k=k)
            kmeans_clusters.evaluate_cluster(observed=df_pca[y_label],y_label=y_label, predicted=cluster_obj.labels_, k=k)
            kmeans_clusters.plot_3dScatter(k,save=True)

            cluster_obj = agglomerative_clustering(df=df_pca,features=features,k=k)
            hac_clusters.evaluate_cluster(observed=df_pca[y_label],y_label=y_label, predicted=cluster_obj.labels_, k=k)
            hac_clusters.plot_3dScatter(k,save=True)


        kmeans_clusters.get_results(plot=False, save=True)
        hac_clusters.get_results(plot=False, save=True)


        """Applying DBSCAN to our dataset"""
        cluster_obj = DBSCAN_clustering(df=df_pca,features=features)
        dbscan_cluster.evaluate_cluster(observed=df_pca[y_label],y_label = y_label, predicted=cluster_obj.labels_)
        dbscan_cluster.get_results(plot=False,save=True)
        dbscan_cluster.plot_3dScatter(save=True)
