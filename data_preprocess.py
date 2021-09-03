import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_missing_null_nan(df):
    """Find out how many null, missing or nan values we have"""

    for y,col in enumerate(df.columns):
        missing_values = np.size(np.where(pd.isnull(df[col])))
        print('{} : {} / {} are null,missing or nan'.format(col, missing_values, len(df)))

def remove_missing(df):
    """Drop every row that contains either a missing, null or nan value"""

    df = df.dropna()
    #no_value_rows = [i for i,v in enumerate(df[col]) if v=='']
    return df


def remove_duplicates(df):
    duplicates = np.sum(df.duplicated(subset=None, keep='first'))
    print("Found {} duplicates in the {} data.\n".format(duplicates,len(df)))
    df = df.drop_duplicates()
    print("Data left: {}".format(len(df)))
    return df

def boxplot_features(df):
    for col in df.columns:
        print('\nStats for class label: {}.'.format(col))
        print(df[col].describe())
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(df[col])
        ax.set_title(col)
        plt.show()

def remove_outliers(df):
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
        print("ABOVE MAX: {}\nBELOW MIN: {}".format(upper_fence_sum,lower_fence_sum))
        upper_fence_sum_quartile = upper_fence_sum/len(df)
        lower_fence_sum_quartile = lower_fence_sum/len(df)
        return lower_fence_sum_quartile,1-upper_fence_sum_quartile

    for col in df.columns:
        min_q, max_q = get_thresholds(df,col)
        print("Outliers: {}\n".format([min_q,max_q]))
        quartile_max = df[col].quantile(max_q)
        quartile_min = df[col].quantile(min_q)
        print("Class feature: {}.".format(col))
        print("Number of values greater than {} quantile: {}".format(max_q,np.sum(df[col]>quartile_max)))
        print("Number of values lower than {} quantile: {}".format(min_q,np.sum(df[col]<quartile_min)))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(df[col])
        ax.set_title(col+' BEFORE REMOVING OUTLIERS')
        plt.show()

        print("DF BEFORE: ",len(df[col]))
        df = df[df[col] <= quartile_max]
        df = df[df[col] >= quartile_min]
        print("DF AFTER: {}\n".format(len(df[col])))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(df[col], sym='')
        ax.set_title(col+' AFTER REMOVING OUTLIERS')
        plt.show()




if __name__ == '__main__':
    """Loading the data"""
    df = pd.read_excel('Data/OriginalCTG.xls','Data')
    df = df.iloc[:,10:31]
    #df = pd.DataFrame(df.iloc[1:,:], columns=df.iloc[0,:])
    df.columns = df.iloc[0,:]
    df = df.drop(df.index[0])

    """Printing and removing missing, null and nan values"""
    #get_missing_null_nan(df)
    df = remove_missing(df)
    # get_missing_null_nan(df)

    """Removing duplicated values"""
    df = remove_duplicates(df)

    """Find and remove outliers"""
    remove_outliers(df)


    #df['LB'].hist(bins=100, edgecolor='black')

    #boxplot_features(df)
