import csv

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, OPTICS, Birch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix, fowlkes_mallows_score, normalized_mutual_info_score,completeness_score


def get_data():
    # get_data()
    with open('pre_proccess_data.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    data.pop(0)
    print(data)
    return data

def split_data_X_Y(data):
    X = []
    Y = []
    for i in range(0, len(data)):
        X.append([x for _, x in zip(range(4), data[i])])
        Y.append(data[i][-10:])
    return X, Y


def round_labels(labels):
    labels = np.array(labels)
    round_label = []
    for i in range(0,len(labels)):
        round_label.append(np.argmax(labels[i], axis=0))
    return round_label

def print_feature_info(fn,tn,x,y):
    print("Feature names: ", ['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc'])
    print("Target names: ", ['CYT', 'NUC', 'MIT', 'ME3', 'ME2', 'ME1', 'EXC', 'VAC', 'POX', 'ERL'])
    print("Sample of input: ", data_X[:2, :])
    print("Sample of output: ", data_Y[:2])

def create_2d_plot(X,labels,name):
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.title(name)
    plt.show()

def create_3d_plot(X,labels,name):
    fig = plt.figure()  # make 3d fig
    ax = Axes3D(fig)
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels, edgecolor='k')
    ax.set_title(name)
    fig.show()

def get_confusion_matrix(original_labels,predicted_labels):
    conmat = confusion_matrix(original_labels, predicted_labels)
    print("Confusion Matrix:\n", conmat)
    return conmat

def metrics(Y,labels):
    print('Y :{}\nLabels: {}'.format(Y,labels))
    print("Fowlkes mallows score:",fowlkes_mallows_score(Y, labels))
    print("Completeness score:",completeness_score(Y, labels))

def precentage(lab,tlab):
    sum = 0
    for i in range(len(lab)):
        if lab[i] == tlab[i]:
            sum = sum + 1
    return sum/len(lab)

def clustering_k_means(X,Y):
    kmeans = KMeans(n_clusters=10, init='k-means++', n_init=10, max_iter=300,tol=0.0001, precompute_distances='auto', verbose=0, random_state=5,copy_x=True, n_jobs=None, algorithm='auto')
    kmeans = kmeans.fit(X)  # fit the model
    labels = kmeans.labels_
    print("predictions: ", labels)
    print(labels)
    print("original: ", Y)
    get_confusion_matrix(Y,labels)
    metrics(Y,labels)
    input()
    create_2d_plot(X,labels,"K-MEANS Protein")
    create_3d_plot(X,data_Y,"Original labels K-MEAN")
    create_3d_plot(X,labels,"Predicted labels K-MEAN")
    print("Percentage of correct labels:",precentage(labels,Y))

def clustering_dbscan(X,Y):
    dbscan = DBSCAN(eps=5,min_samples=2,metric='euclidean',algorithm='auto',leaf_size=40,p=2,n_jobs=None)  # define the model
    dbscan = dbscan.fit(X)  # fit the model
    labels = dbscan.labels_
    print("predictions: ", labels)
    print(labels)
    print("original: ", Y)
    get_confusion_matrix(Y,labels)
    metrics(Y,labels)
    create_2d_plot(X,labels,"DBSCAN Protein")
    create_3d_plot(X,data_Y,"Original labels DBSCAN")
    create_3d_plot(X,labels,"Predicted labels DBSCAN")
    print("Percentage of correct labels:",precentage(labels,Y))

def clustering_birch(X,Y):
    birch = Birch(n_clusters=10,threshold=1.1,branching_factor=20,compute_labels=True,copy=True) # define the model
    birch = birch.fit(X)  # fit the model
    labels = birch.labels_
    print("predictions: ", labels)
    print(labels)
    print("original: ", Y)
    get_confusion_matrix(Y,labels)
    metrics(Y,labels)
    create_2d_plot(X,labels,"BIRCH Protein")
    create_3d_plot(X,data_Y,"Original labels BIRCH")
    create_3d_plot(X,labels,"Predicted labels BIRCH")
    print("Percentage of correct labels:",precentage(labels,Y))

data = get_data()
data_X, data_Y = split_data_X_Y(data)
data_Y = round_labels(data_Y)
data_X = np.array(data_X)
data_Y = np.array(data_Y)
#print_feature_info(['mcg','gvh','alm','mit','erl','pox','vac','nuc'],['CYT','NUC','MIT','ME3','ME2','ME1','EXC','VAC','POX','ERL'],data_X,data_Y)

#create_2d_plot(data_X,data_Y,"Original Protein")

data_X = StandardScaler().fit_transform(data_X ) # Standarize features

#K-MEAN ALGORITHM
print("---------------","K-MEANS ALGORITHM","---------------")
clustering_k_means(data_X,data_Y)
print("---------------","DBSCAN ALGORITHM","---------------")
clustering_dbscan(data_X,data_Y)
print("---------------","BIRCH ALGORITHM","---------------")
clustering_birch(data_X,data_Y)
