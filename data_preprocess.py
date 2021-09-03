import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
    TO-DO:
        Z-SCORE=DONE,
        STANDARD-DEVIATION,
        STANDARIZATION-NORMALIZATION,
        PCA,
        SAVE PLOTS IN A FOLDER
"""
def get_missing_null_nan(df,remove=False):
    """Find out how many null, missing or nan values we have. Drop them if remove==True"""
    for y,col in enumerate(df.columns):
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

def boxplot_features(df):
    for col in df.columns:
        print('\nStats for class label: {}.'.format(col))
        print(df[col].describe())
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(df[col])
        ax.set_title('Feature: '+col)
        plt.show()

def hist_features(df):
    """Plot the histograms of the all features"""
    for col in df.columns:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(df[col], bins=100, edgecolor='black')
        ax.set_title('Feature: '+col)
        ax.set_xlabel('Range of Values')
        ax.set_ylabel('Num of Observations')
        plt.show()

def quantile_outliers(df, remove=False, plot=False):
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
    for col in df.columns:
        min_q, max_q = get_thresholds(df,col)
        quartile_max = df[col].quantile(max_q)
        quartile_min = df[col].quantile(min_q)
        print("\nClass feature: {}.".format(col))
        print("Number of values greater than {} quantile: {}".format(max_q,np.sum(df[col]>quartile_max)))
        print("Number of values lower than {} quantile: {}".format(min_q,np.sum(df[col]<quartile_min)))

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.boxplot(df[col])
            ax.set_title(col+' BEFORE REMOVING OUTLIERS')

        if remove:
            print("OBSERVATIONS BEFORE: ",len(df[col]))
            df = df[df[col] <= quartile_max]
            df = df[df[col] >= quartile_min]
            print("OBSERVATIONS AFTER: {}\n".format(len(df[col])))

            if plot:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.boxplot(df[col], sym='')
                ax.set_title(col+' AFTER REMOVING OUTLIERS')
        plt.show()

def z_score_outliers(df, k=3, remove=False):
    """Returns stats about the outliers detected based on z_score. Drops them if remove==True"""
    for col in df.columns:
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





if __name__ == '__main__':
    """Loading the data"""
    df = pd.read_excel('Data/OriginalCTG.xls','Data')
    df = df.iloc[:,10:31]
    df.columns = df.iloc[0,:]
    df = df.drop(df.index[0])

    print(len(df))

    """Printing and removing missing, null and nan values"""
    #get_missing_null_nan(df)
    df = get_missing_null_nan(df, remove=True)
    # get_missing_null_nan(df)

    """Removing duplicated values"""
    df = remove_duplicates(df)

    """Find and remove outliers"""
    #quantile_outliers(df,remove=True, plot=False)

    #hist_features(df)
    #df['LB'].hist(bins=100, edgecolor='black')
    z_score_outliers(df, remove=True)
    #hist_features(df)
