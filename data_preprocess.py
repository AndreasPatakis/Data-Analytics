import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_missing_null_nan(df):
    """Find out how many null, missing or nan values we have"""

    for y,col in enumerate(df.columns):
        missing_values = np.size(np.where(pd.isnull(df[col])))
        print('{} : {} / {} are null,missing or nan'.format(df.iloc[0,y], missing_values, len(df.iloc[1:,:])))

def remove_missing(df):
    """Drop every row that contains either a missing, null or nan value"""

    df = df.dropna()
    #no_value_rows = [i for i,v in enumerate(df[col]) if v=='']
    return df






if __name__ == '__main__':
    #Reading  our data
    df = pd.read_csv('Data/CTGData.csv')
    df = df.iloc[:,10:31]

    #Some information about missing, null or nan values
    # get_missing_null_nan(df)
    # df = remove_missing(df)
    # get_missing_null_nan(df)

    df['1'].hist(bins=100)
    plt.show()
