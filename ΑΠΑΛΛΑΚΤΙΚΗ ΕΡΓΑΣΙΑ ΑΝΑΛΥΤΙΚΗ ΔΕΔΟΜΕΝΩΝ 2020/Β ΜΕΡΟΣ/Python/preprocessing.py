import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
from pandas import Grouper, DataFrame
from pandas.plotting import lag_plot, autocorrelation_plot
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def draw_centralTedency(dataframe):
    fig = plt.figure()
    gs = gridspec.GridSpec(8, 8)
    ax = plt.gca()
    mean = dataframe.mean().values[0]
    median = dataframe.median().values[0]
    mode = dataframe.mode().values[0]
    sns.distplot(dataframe, ax=ax)
    ax.axvline(mean, color='r', linestyle='--')
    ax.axvline(median, color='g', linestyle='-')
    ax.axvline(mode, color='b', linestyle='-')
    plt.legend({'Mean': mean, 'Median': median, 'Mode': mode})
    plt.show()


def barplot_features_with_date(dataset:DataFrame, columns):
    pos = np.arange(len(dataset['date']))

    for column in columns:
        if column == 'USD ISE':
            continue
        _, ax = plt.subplots()
        dataset.plot(kind='bar', x=dataset.columns.get_loc("date"), y=dataset.columns.get_loc(column), color='blue',
                     ax=ax)
        dataset.plot(kind='bar', x=dataset.columns.get_loc("date"), y=dataset.columns.get_loc('USD ISE'), color='red',
                     ax=ax)
        ticks = plt.xticks(pos[::15], rotation=90)
        plt.rcParams["figure.figsize"] = (15, 8)
        plt.show()


def timeseries_to_supervised(df, n_in, n_out):
    agg = pd.DataFrame()
    for i in range(n_in, 0, -1):
        df_shifted = df.shift(i).copy()
        df_shifted.rename(columns=lambda x: ('%s(t-%d)' % (x, i)), inplace=True)
        agg = pd.concat([agg, df_shifted], axis=1)

    for i in range(0, n_out):
        df_shifted = df.shift(-i).copy()
        if i == 0:
            df_shifted.rename(columns=lambda x: ('%s(t)' % (x)), inplace=True)
        else:
            df_shifted.rename(columns=lambda x: ('%s(t+%d)' % (x, i)), inplace=True)
        agg = pd.concat([agg, df_shifted], axis=1)
    agg.dropna(inplace=True)
    return agg

#if show_plots is true the plots are drawn.
def preprocess(dataset: DataFrame, n_in: int, n_out: int, show_plots: bool):
    # plot the data to get insights
    dataset.reset_index(inplace=True, drop=True)
    dataset = dataset.iloc[0:530]
    # print(dataset['USD ISE'])
    columns = ['TL ISE', 'USD ISE', 'SP', 'DAX', 'FTSE', 'NIKEEI', 'BOVESPA', 'EU', 'EM']
    if show_plots == True:

        dataset.boxplot()
        plt.show()
        plt.plot(dataset['USD ISE'])
        dataset.hist()
        plt.show()
        ax = plt.gca()
        dataset.plot(kind='line', x=0, y=3, color='blue', ax=ax)
        dataset.plot(kind='scatter', x=0, y=3, color='red', ax=ax)
        plt.show()


        for column in columns:
          #  figure(1, figsize=(6, 6))
            lag_plot(dataset[column].to_frame())
            #plt.savefig(column+'.png')
            plt.xlabel(column + ' (t)')
            plt.ylabel(column + ' (t + 1)')
            plt.show()

        for column in columns:
            autocorrelation_plot(dataset[column])
            plt.xlabel(column + ' Autocorrelation')
            plt.ylabel(column + ' Lag')
            plt.show()

# delete the date column from the dataset
    del dataset['date']

    #feature_scaler = MinMaxScaler(feature_range=(-1, 1))
    index_scaler = MinMaxScaler(feature_range=(-1, 1))

    print(dataset.shape)

    stock = dataset['USD ISE'].to_frame()
    scaled_stock = index_scaler.fit_transform(np.reshape(stock['USD ISE'].values, (stock.shape[0], stock.shape[1])))
    scaled_stock = pd.DataFrame(data=scaled_stock, columns=['USD ISE'])

    features = dataset.copy()
    del features['USD ISE']
    del features['TL ISE']
    # delete the TL ISE index as we are going to focus on USD ISE, and delete the
    # USD ISE index because we wont use it as a feature of the vector X.
    #scaled_features = feature_scaler.fit_transform(np.reshape(features.values, (features.shape[0], features.shape[1])))

    #feature_columns = ['SP', 'DAX', 'FTSE', 'NIKEEI', 'BOVESPA', 'EU', 'EM']
    #scaled_features = pd.DataFrame(data=scaled_features, columns=feature_columns)


    superv_stock = timeseries_to_supervised(scaled_stock, n_in, n_out)
    superv_features = timeseries_to_supervised(features, n_in, n_out)

    # Dataset  transformed to supervised problem of series (samples) 30 composed of
    # 1 days is of size [samples = the rows of any of our dataframe, 1 (for each day) ,
    # features = the columns of the features dataframe]:
    samples = superv_stock.shape[0]
    features = superv_features.shape[1]
    steps = 1
    # returns the scalers too so that we are able to get the inverted vectors of
    # the dataframes that we scaled
    return superv_stock, superv_features, [samples, steps, features], index_scaler


# file_name = 'data_akbilgic.xlsx'
# data = pd.read_excel(file_name, header=None, skiprows=2)
# data.dropna(inplace=True)
# data.columns = ['date', 'TL ISE', 'USD ISE', 'SP', 'DAX', 'FTSE', 'NIKEEI', 'BOVESPA', 'EU', 'EM']
#preprocess(data, 1, 1, True)
