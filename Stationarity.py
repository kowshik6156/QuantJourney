'''
This file part of my quant journey series 

In this project, we will explore monthly returns and timeseries charts. We will apply differencing to
remove seasonality. Then we will do a test for stationarity and use ARIMA model for differencing and conversion of
non-stationary series to stationary (removing the trend and seasonality)

For the ones who are not aware of time series models,
https://qr.ae/pGCSKN

'''



import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

start_date = '2001-01-01'
stock_data = data.DataReader('GOOG', 'yahoo', start_date)
stock_monthly_return = []
stock_monthly_return = stock_data['Adj Close'].pct_change().groupby(
    [stock_data['Adj Close'].index.year, stock_data['Adj Close'].index.month]).mean()

stock_monthly_return_list = []

for i in range(len(stock_monthly_return)):
    stock_monthly_return_list.append(
        {'month': stock_monthly_return.index[i][1], 'monthly_return': stock_monthly_return.values[i]})
stock_monthly_return_list = pd.DataFrame(stock_monthly_return_list, columns=('month', 'monthly_return'))

stock_monthly_return_list.boxplot(column='monthly_return', by='month')
ax = plt.gca()
labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticklabels(labels)
ax.set_ylabel('GOOG Return')
plt.tick_params(axis='both', which='major', labelsize=7)
plt.title("GOOG Monthly Return")
plt.suptitle("")
plt.show()



# Checking for stationarity (mean, variance remain constant over time)
def plot_rolling_statistics(ts, titletext, ytext, window_size):
    ts.plot(color='r', label='Original', lw=0.5)
    ts.rolling(window_size).mean().plot(color='b', label='Rolling Mean')
    ts.rolling(window_size).std().plot(color='g', label='Rolling Std')

    plt.legend(loc='best')
    plt.ylabel(ytext)
    plt.title(titletext)
    plt.show()


plot_rolling_statistics(stock_monthly_return[1:], 'GOOG Monthly Return, Rolling Mean & Std', 'Monthly Return', 12)
plot_rolling_statistics(stock_data['Adj Close'], 'GOOG Prices, Rolling Mean and Std', 'Daily Prices', 365)

# to remove the trend, simply subtract the daily prices with moving average. We are basically removing seasonality
plot_rolling_statistics(stock_data['Adj Close']-stock_data['Adj Close'].rolling(365).mean(),
                        'GOOG prices without trend', 'Daily prices', 365)

# Dickey-Fuller test
# Stationarity means mean and variance remain constant over time
# To determine the presence of a unit root in time series
# if unit root is present, then the time series is not stationary
# Null hypothesis of this test is that test has a unit root.
# if we reject null hypothesis, this means that we don't find a unit root
# if we fail to reject the null hypothesis, we can say time series is non stationary

def test_stationarity(timeseries):
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries[1:], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    print(dfoutput)

test_stationarity(stock_monthly_return[1:])
test_stationarity(stock_data['Adj Close'])  # if pvalue is > 0.05, we cannot reject the null hypothesis

# forcasting time series
# For a stationarity series without dependencies, we can use regular linear regression to forecast values
# A series with dependencies among values (non-stationary), we will use statistical models such as ARIMA
# parameter values for AR and MA can be found using ACF and PACF respectively
# differencing can be applied one or more times to eliminate the non-stationarity of the mean function (trend)


pyplot.figure()
pyplot.subplot(211)
plot_acf(stock_monthly_return[1:], ax=pyplot.gca(), lags=10)
pyplot.subplot(212)
plot_pacf(stock_monthly_return[1:], ax=pyplot.gca(), lags=10)
pyplot.show()
# from the above plot, we can say that the order of AR and MA is lag 1 as the ACF and PACF chart crosses the
# upper confidence

model = ARIMA(stock_monthly_return[1:], order=(2,0,2))
fitted_results = model.fit()
stock_monthly_return[1:].plot()
fitted_results.fittedvalues.plot(color='red', title='ARIMA')
plt.show()





