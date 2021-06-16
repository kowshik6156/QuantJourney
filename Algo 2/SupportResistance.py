# This is part of my quant journey series 
# Second basic trading strategy
# buy at support and sell at resistance
# logic is explained along the code





import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas_datareader import data
import matplotlib.pyplot as plt

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

start_date = '2014-01-02'
end_date = '2018-01-01'
goog_data = data.DataReader('GOOG', 'yahoo', start_date, end_date)


goog_data_signal = pd.DataFrame(index=goog_data.index)
goog_data_signal['price'] = goog_data['Adj Close']


def trading_support_resistance(data, bin_width):
    data['sup_tolerance'] = pd.Series(np.zeros(len(data)))
    data['res_tolerance'] = pd.Series(np.zeros(len(data)))
    data['sup_count'] = pd.Series(np.zeros(len(data)))
    data['res_count'] = pd.Series(np.zeros(len(data)))
    data['sup'] = pd.Series(np.zeros(len(data)))
    data['res'] = pd.Series(np.zeros(len(data)))
    data['positions'] = pd.Series(np.zeros(len(data)))
    data['signal'] = pd.Series(np.zeros(len(data)))
    in_support = 0
    in_resistance = 0

    for x in range((bin_width - 1) + bin_width, len(data)):   # range starts from 39 to end of the trading data
        data_section = data[x - bin_width:x + 1]
        # This is the window. For example, when x=39, the window range is 19 to 39.
        # This range is used to find min and max values of the range 19 to 39 to set the support and resistance at 39
        # min value acts as support and max value acts as resistance
        # range is the difference between min and max

        support_level = min(data_section['price'])
        resistance_level = max(data_section['price'])
        range_level = resistance_level - support_level
        data['res'][x] = resistance_level
        data['sup'][x] = support_level

        # assume tolerances as bands within support and resistance set in line 46 and 47

        data['sup_tolerance'][x] = support_level + 0.2 * range_level
        data['res_tolerance'][x] = resistance_level - 0.2 * range_level

        if data['price'][x] >= data['res_tolerance'][x] and data['price'][x] <= data['res'][x]:
            in_resistance += 1
            data['res_count'][x] = in_resistance
        elif data['price'][x] <= data['sup_tolerance'][x] and data['price'][x] >= data['sup'][x]:
            in_support += 1
            data['sup_count'][x] = in_support
        else:
            in_support = 0
            in_resistance = 0

        # we send buy order if the price is between support and sup_tolerance (signal is 1)
        # we send sell order if the price is between resistance and res_tolerance (signal is 0)

        if in_resistance > 2:
            data['signal'][x] = 0
        elif in_support > 2:
            data['signal'][x] = 1
        else:
            data['signal'][x] = data['signal'][x-1]

    # similar to first strategy, position stores 1 and -1 (buy and sell)
    data['positions'] = data['signal'].diff()

    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel='Google price in $')
    data['sup'].plot(ax=ax1, color='g', lw=2.)
    data['res'].plot(ax=ax1, color='b', lw=2.)
    data['price'].plot(ax=ax1, color='r', lw=2.)
    ax1.plot(data.loc[data.positions == 1.0].index, data.price[data.positions == 1.0], '^', markersize=7, color='k',label='buy')
    ax1.plot(data.loc[data.positions == -1.0].index, data.price[data.positions == -1.0], 'v', markersize=7, color='k',label='sell')
    plt.legend()
    plt.show()


trading_support_resistance(goog_data_signal, 20)

