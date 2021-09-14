import csv
import sys as os
import urllib.request
import numpy as np
import pandas as pd, scipy, numpy
from sklearn.preprocessing import MinMaxScaler
import numpy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def get_data():
    urllib.request.urlretrieve("http://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data", "yeastdata.csv")

def remove_accession_number(tdf):
    tdf = tdf.drop(columns=['Sequence Name'])
    return tdf
    
def empty_cell_lines(tdf):
    print(tdf.isnull().sum(axis=1))
    return tdf.isnull().sum(axis=1).values.sum()

def num_class_labels(data):
    print(df['Name'].value_counts())

def data_value_range(tdf):
    tdf = tdf.drop(columns=['Name'])
    print(tdf.max(axis=0))
    print(tdf.min(axis=0))
    return min(tdf.min(axis=0)),max(tdf.max(axis=0))

def plots(tdf):
    tdf = tdf.drop(columns=['Name'])
    print(tdf)
    tdf.hist(bins=12, alpha=0.5)
    plt.show()
    df.plot(kind='density', subplots=True, sharex=False)
    plt.show()
    tdf.plot(kind='box', subplots=True, sharex=False, sharey=False)
    plt.show()
    correlations = tdf.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=0, vmax=1)
    fig.colorbar(cax)
    ticks = numpy.arange(0, 9, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    plt.show()
    pd.plotting.scatter_matrix(tdf)
    plt.show()

def dublicated(tdf):
    tdf.drop_duplicates(keep=False)
    print("Removed dublicates",df.shape)
    return tdf

def encode_names(tdf):
    tdf['Name'] = pd.Categorical(tdf['Name'])
    dfDummies = pd.get_dummies(tdf['Name'], prefix='category')
    tdf = pd.concat([tdf, dfDummies], axis=1)
    tdf = tdf.drop(['Name'], axis=1)
    print(tdf)
    return tdf

def PCA_reduction(tdf):
    pca = PCA(n_components=4)
    principalComponents = pca.fit_transform(df[['mcg','gvh','alm','mit','erl','pox','vac','nuc']])
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2','principal component 3','principal component 4'])
    finalDf = pd.concat([principalDf, tdf[['Name']]], axis=1)
    print(finalDf)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('4 component PCA', fontsize=20)
    img = ax.scatter(finalDf['principal component 1'],
                         finalDf['principal component 2'],
                         finalDf['principal component 3'],
                         c=finalDf['principal component 4'], cmap=plt.hot())
    fig.colorbar(img)
    plt.show()
    return finalDf

def random_sampling(tdf):
    tdf = tdf.sample(frac=.25)
    print(tdf)

def export_csv(tdf):
    tdf.to_csv('pre_proccess_data.csv', index=False, header=True)


get_data()
with open('yeastdata.csv') as input_file:
    lines = input_file.readlines()
    newLines = []
    for line in lines:
        newLine = line.strip().split()
        newLines.append(newLine)


display = pd.options.display
display.max_columns = 1000000
display.max_colwidth = 10000000
display.width = None

df = pd.DataFrame(newLines, columns =['Sequence Name','mcg','gvh','alm','mit','erl','pox','vac','nuc','Name'])
df['mcg'] = df['mcg'].astype(float)
df['gvh'] = df['gvh'].astype(float)
df['alm'] = df['alm'].astype(float)
df['mit'] = df['mit'].astype(float)
df['erl'] = df['erl'].astype(float)
df['pox'] = df['pox'].astype(float)
df['vac'] = df['vac'].astype(float)
df['nuc'] = df['nuc'].astype(float)
start_df = df

print("The shape of the data",df.shape)
print("Lines with empty cell(s): "+str(empty_cell_lines(df)))
print("False labels:",num_class_labels(df))
df = remove_accession_number(df)
print(df)
dublicated(df)
minn,maxx = data_value_range(df)
print("Min data value:",minn)
print("Max data value:",maxx)
random_sampling(df)
scaler = StandardScaler()
df[['mcg','gvh','alm','mit','erl','pox','vac','nuc']] = scaler.fit_transform(df[['mcg','gvh','alm','mit','erl','pox','vac','nuc']])
df = PCA_reduction(df)
scaler = MinMaxScaler()
df[['principal component 1', 'principal component 2','principal component 3','principal component 4']] = scaler.fit_transform(df[['principal component 1', 'principal component 2','principal component 3','principal component 4']])
plots(start_df)
plots(df)
df = encode_names(df)
export_csv(df)




