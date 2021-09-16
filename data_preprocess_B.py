import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from pandas import DataFrame
from pandas.plotting import lag_plot, autocorrelation_plot
from sklearn.preprocessing import MinMaxScaler


def draw_centralTdn(dataframe):
    fig = plot.figure()
    gs = gridspec.GridSpec(8, 8)
    ax = plot.gca()
    mean = dataframe.mean().values[0]
    median = dataframe.median().values[0]
    mode = dataframe.mode().values[0]
    sns.distplot(dataframe, ax=ax)
    ax.axvline(mean, color='r', linestyle='--')
    ax.axvline(median, color='g', linestyle='-')
    ax.axvline(mode, color='b', linestyle='-')
    plot.legend({'Mean': mean, 'Median': median, 'Mode': mode})
    plot.show()


def barplot_ft_with_date(dataset: DataFrame, columns):
    pos = np.arange(len(dataset['date']))

    for column in columns:
        if column == 'USD ISE':
            continue
        _, ax = plot.subplots()
        dataset.plot(kind='bar', x=dataset.columns.get_loc("date"), y=dataset.columns.get_loc(column), color='blue',
                     ax=ax)
        dataset.plot(kind='bar', x=dataset.columns.get_loc("date"), y=dataset.columns.get_loc('USD ISE'), color='red',
                     ax=ax)
        ticks = plot.xticks(pos[::15], rotation=90)
        plot.rcParams["figure.figsize"] = (15, 8)
        plot.show()


def timeseries_to_supervised(df, n_in, n_out):
    ag = pd.DataFrame()
    for i in range(n_in, 0, -1):
        df_shifted = df.shift(i).copy()
        df_shifted.rename(columns=lambda x: ('%s(t-%d)' % (x, i)), inplace=True)
        ag = pd.concat([ag, df_shifted], axis=1)

    for i in range(0, n_out):
        df_shifted = df.shift(-i).copy()
        if i == 0:
            df_shifted.rename(columns=lambda x: ('%s(t)' % x), inplace=True)
        else:
            df_shifted.rename(columns=lambda x: ('%s(t+%d)' % (x, i)), inplace=True)
        ag = pd.concat([ag, df_shifted], axis=1)
    ag.dropna(inplace=True)
    return ag


""" If show_plots is true then the plots are drawn """
def preprocess_data(dataset: DataFrame, n_in: int, n_out: int, show_plots: bool):
    """ Plot the data to get insights """
    dataset.reset_index(inplace=True, drop=True)
    dataset = dataset.iloc[0:530]
    columns = ['TL ISE', 'USD ISE', 'SP', 'DAX', 'FTSE', 'NIKEEI', 'BOVESPA', 'EU', 'EM']
    if show_plots:

        dataset.boxplot()
        plot.show()
        plot.plot(dataset['USD ISE'])
        dataset.hist()
        plot.show()
        ax = plot.gca()
        dataset.plot(kind='line', x=0, y=3, color='blue', ax=ax)
        dataset.plot(kind='scatter', x=0, y=3, color='red', ax=ax)
        plot.show()

        lag_plot(dataset['USD ISE'])
        plot.show()
        autocorrelation_plot(dataset['USD ISE'])

        for column in columns:
            lag_plot(dataset[column].to_frame())
            plot.xlabel(column + ' (t)')
            plot.ylabel(column + ' (t + 1)')
            plot.show()

        for column in columns:
            autocorrelation_plot(dataset[column])
            plot.xlabel(column + ' Autocorrelation')
            plot.ylabel(column + ' Lag')
            plot.show()

    """ Delete the date column from the dataset """
    del dataset['date']

    index_scaler = MinMaxScaler(feature_range=(-1, 1))

    print(dataset.shape)

    stock = dataset['USD ISE'].to_frame()
    scaled_st = index_scaler.fit_transform(np.reshape(stock['USD ISE'].values, (stock.shape[0], stock.shape[1])))
    scaled_st = pd.DataFrame(data=scaled_st, columns=['USD ISE'])

    """ Delete the TL ISE index as we are going to focus on USD ISE, and delete the USD ISE index because we wont use it as a feature of the vector X. """
    features = dataset.copy()
    del features['USD ISE']
    del features['TL ISE']

    superv_st = timeseries_to_supervised(scaled_st, n_in, n_out)
    superv_ft = timeseries_to_supervised(features, n_in, n_out)

    """ Dataset transformed to supervised problem of series (samples) 30 composed of 1 days is of size
        [samples = the rows of any of our dataframe, 1 (for each day) , features = the columns of the features dataframe]: """
    samples = superv_st.shape[0]
    features = superv_ft.shape[1]
    steps = 1

    """ Returns the scalers too so that we are able to get the inverted vectors of the dataframes that we scaled """
    return superv_st, superv_ft, [samples, steps, features], index_scaler


""" Run code here """
file_name = 'data_akbilgic.xlsx'
data = pd.read_excel(file_name, header=None, skiprows=2)
data.dropna(inplace=True)
data.columns = ['date', 'TL ISE', 'USD ISE', 'SP', 'DAX', 'FTSE', 'NIKEEI', 'BOVESPA', 'EU', 'EM']
preprocess_data(data, 1, 1, True)
