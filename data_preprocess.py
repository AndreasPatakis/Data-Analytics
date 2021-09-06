import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


"""
    TO-DO:
        Z-SCORE=DONE,
        STANDARIZATION-NORMALIZATION=DONE,
        PCA,
        SAVE PLOTS IN A FOLDER=DONE
"""

"""CONSTANTS"""
BEFORE = '_WITH_OUTLIERS'
AFTER = '_WITHOUT_OUTLIERS'
parent_dir = './Preprocess_Plots'

def get_missing_null_nan(df,remove=False):
    """Find out how many null, missing or nan values we have. Drop them if remove==True"""
    for col in df.columns:
        missing_values = np.size(np.where(pd.isnull(df[col])))
        print('{} : {} / {} are null,missing or nan'.format(col, missing_values, len(df)))
    if remove:
        df = df.dropna()
    return df

def remove_duplicates(df):
    duplicates = np.sum(df.duplicated(subset=None, keep='first'))
    print("Found {} duplicates in the {} data.".format(duplicates,len(df)))
    df = df.drop_duplicates()
    print("Data left: {}\n".format(len(df)))
    return df

def boxplot_features(df,features):
    for col in features:
        print('\nStats for class label: {}.'.format(col))
        print(df[col].describe())
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(df[col])
        ax.set_title('Feature: '+col)
        plt.show()

def hist_features(df,features, plot=True,save=False):
    """Plot the histograms of the all features"""

    if save:
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

    for col in features:
        path = '/'+col+'_HIST'
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(df[col], bins=100, edgecolor='black')
        ax.set_title('Feature: '+col)
        ax.set_xlabel('Range of Values')
        ax.set_ylabel('Num of Observations')
        if save:
            print("Saving histogram of feature: ",col)
            plt.savefig(parent_dir+'/'+col+path)
        if plot:
            plt.show()

def quantile_outliers(df,features, remove=False, plot=False, save=False):
    def get_IQR(df,col):
        """Returns the Interquartile Range of our set"""
        first_quartile = df[col].quantile(0.25)
        third_quartile = df[col].quantile(0.75)
        iqr = third_quartile - first_quartile
        return iqr

    def get_thresholds(df,col):
        """Return Upper_Fence and Lower_Fence as thresholds for detecting outliers"""
        iqr = get_IQR(df,col)
        upper_fence_sum = np.sum(df[col] > df[col].quantile(0.75) + iqr*1.5)
        lower_fence_sum = np.sum(df[col] < df[col].quantile(0.25) - iqr*1.5)
        upper_fence_sum_quartile = upper_fence_sum/len(df)
        lower_fence_sum_quartile = lower_fence_sum/len(df)
        return lower_fence_sum_quartile,1-upper_fence_sum_quartile

    """Removing outliers greater than Upper_Fence and lower than Lower_Fence"""
    if save:
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
    for col in features:
        min_q, max_q = get_thresholds(df,col)
        quartile_max = df[col].quantile(max_q)
        quartile_min = df[col].quantile(min_q)
        print("\nClass feature: {}.".format(col))
        print("Number of values greater than {} quantile: {}".format(max_q,np.sum(df[col]>quartile_max)))
        print("Number of values lower than {} quantile: {}".format(min_q,np.sum(df[col]<quartile_min)))

        if plot or save:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.boxplot(df[col])
            ax.set_title(col+BEFORE)
            feature_folder = os.path.join(parent_dir+'/'+col)
            if save:
                if not os.path.exists(feature_folder):
                    os.makedirs(feature_folder)
                feature_folder_path = feature_folder+'/'+col
                plt.savefig(feature_folder_path+BEFORE)

        if remove:
            print("OBSERVATIONS BEFORE: ",len(df[col]))
            df = df[df[col] <= quartile_max]
            df = df[df[col] >= quartile_min]
            print("OBSERVATIONS AFTER: {}\n".format(len(df[col])))

            if plot or save:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.boxplot(df[col], sym='')
                ax.set_title(col+AFTER)
                if save:
                    plt.savefig(feature_folder_path+AFTER)
                if plot:
                    plt.show()
    return df

def z_score_outliers(df,features, k=3, remove=False):
    """Returns stats about the outliers detected based on z_score. Drops them if remove==True"""
    for col in features:
        print("\nFor feature {}: ".format(col))
        mean = df[col].mean()
        std = df[col].std(axis=0, skipna=True)
        df['z_score'] = (df[col]-mean) / std
        print("Standard Deviation: ",std)
        print('z_score:\n',df[[col,'z_score']])
        print('Observations greater than {} times the standard deviation: {}'.format(k,len(df[df['z_score']>k][col])))
        print('Observations smaller than {} times the standard deviation: {}'.format(k,len(df[df['z_score']<-k][col])))

        if remove:
            print("Num of Observations before removing outliers: ",len(df))
            df = df[df['z_score']>-k]
            df = df[df['z_score']<k]
            df = df.drop(['z_score'], axis=1)
            print("Num of Observations after removing outliers: ",len(df))

def normalize(df,features):
    """Using MinMaxScaler to normalize our data"""
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

def pca(df,features,dimentions=3):
    """Applying Principle Component Analysis to reduce our dimentions to 3"""
    pca = PCA(n_components=dimentions)
    pca = pca.fit(df[features])
    pca_features = pca.transform(df[features])
    return pd.DataFrame(pca_features)


if __name__ == '__main__':
    """Loading the data"""
    df = pd.read_excel('Data/OriginalCTG.xls','Data')
    df = df.iloc[:,10:]
    df.columns = df.iloc[0,:]
    df = df.drop(df.index[0])
    features = ['LB','AC','FM','UC','DL','DS','DP','ASTV','MSTV','ALTV','MLTV','Width','Min','Max','Nmax','Nzeros','Mode','Mean','Median','Variance','Tendency']
    features_with_classes = features + ['CLASS', 'NSP']
    df = df[features_with_classes]


    """Printing and removing missing, null and nan values"""
    df = get_missing_null_nan(df,remove=True)

    """Removing duplicated values"""
    df = remove_duplicates(df)

    """Find and remove outliers"""
    #df = quantile_outliers(df,features=features,remove=True, plot=False, save=True)

    #hist_features(df)
    #df['LB'].hist(bins=100, edgecolor='black')
    #z_score_outliers(df, remove=True)
    #hist_features(df,features=features,plot=False,save=True)

    """Rearange index numbers after deletions"""
    df.index = np.arange(1,len(df)+1)
    df = normalize(df,features=features)

    pca_features = pca(df,features)
    pca_features.index = np.arange(1,len(df)+1)
    pca_features.columns = ['PCA1', 'PCA2', 'PCA3']
    pca_features['CLASS'] = df['CLASS']
    pca_features['NSP'] = df['NSP']



    # pca_features.to_csv('./Data/PCA_Features.csv',index=False, header=True)
    #
    #
    # df.to_csv('./Data/cleanedCTG.csv',index=False, header=True)
