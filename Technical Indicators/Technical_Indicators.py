# This is part of my quant journey series
# This code consists of various technical analysis indicators.
# SMA, EMA, STD, APO, MACD, Bollinger Band, Momentum, RSI
# both charts and calculations are provided
# Time period: From 2018 to previous day (only last trading 500 days will be considered)

import pandas as pd
from pandas_datareader import data
import statistics as stats
import matplotlib.pyplot as plt
import math as math

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

start_date = '2018-01-02'

goog_data = data.DataReader('GOOG', 'yahoo', start_date)
goog_data = goog_data.tail(500)

# changing the index date format YYYY:MM:DD 00:00:00 to YYYY:MM:DD
goog_data.index = pd.to_datetime(goog_data.index).date
goog_data['Date'] = goog_data.index  # creating a new column and storing index values
first_column = goog_data.pop('Date') # moving the date to first column.
goog_data.insert(0, 'Date', first_column)
# Simple Moving Average Calculation
time_period = 20
history = []  # storing values temporarily
sma_values = []  # storing simple moving average values

for close_price in goog_data.Close:
    history.append(close_price)
    if len(history) > time_period:
        del (history[0])  # delete when the history is above 20
    sma_values.append(stats.mean(history))

goog_data = goog_data.assign(Simple20DayMovingAverage=pd.Series(sma_values, index=goog_data.index))

# Exponential Moving Average Calculation
time_period = 20
K = 2 / (time_period + 1)
ema_p = 0  # this variable will store recent close price
ema_values = []  # storing EMA values

for close_price in goog_data.Close:
    if ema_p == 0:
        ema_p = close_price
    else:
        ema_p = close_price * K + ema_p * (1 - K)
    ema_values.append(ema_p)

goog_data = goog_data.assign(Exponential20DayMovingAverage=pd.Series(ema_values, index=goog_data.index))

# chart for SMA and EMA
fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Google Price in $', title="SMA vs EMA vs Close")
goog_data.Close.plot(ax=ax1, color='g', lw=2, legend=True, label="Close Price")
goog_data.Simple20DayMovingAverage.plot(ax=ax1, color='r', lw=2., legend=True, label="SMA")
goog_data.Exponential20DayMovingAverage.plot(ax=ax1, color='b', lw=2., legend=True, label="EMA")
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

# standard deviation
time_period = 20
history = []
sma_values = []
stddev_values = []

for close_price in goog_data.Close:
    history.append(close_price)
    if len(history) > 20:
        del (history[0])

    sma = stats.mean(history)
    sma_values.append(sma)

    variance = 0
    for hist_price in history:
        variance = variance + ((hist_price - sma) ** 2)

    stdev = math.sqrt(variance / len(history))
    stddev_values.append(stdev)

goog_data = goog_data.assign(StandardDeviation20Days=pd.Series(stddev_values, index=goog_data.index))

# Chart for standard deviation
fig = plt.figure()
ax1 = fig.add_subplot(211, ylabel='Google Price in $', title="Standard Deviation")
goog_data.Close.plot(ax=ax1, color='g', lw=2, legend=True, label="Close Price")
ax2 = fig.add_subplot(212, ylabel='Standard Deviation')
goog_data.StandardDeviation20Days.plot(ax=ax2, color='black', lw=2., legend=True, label="Stdev")
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

# Absolute Price Oscillator --> difference between fast EMA and slow EMA
time_period_fast = 10
time_period_slow = 40
K_fast = 2 / (time_period_fast + 1)
K_slow = 2 / (time_period_slow + 1)
ema_fast = 0
ema_slow = 0
ema_fast_values = []
ema_slow_values = []
apo_values = []  # stores computed APO values

for close_price in goog_data.Close:
    if ema_fast == 0:
        ema_fast = close_price
        ema_slow = close_price
    else:
        ema_fast = close_price * K_fast + ema_fast * (1 - K_fast)
        ema_slow = close_price * K_slow + ema_slow * (1 - K_slow)
    ema_fast_values.append(ema_fast)
    ema_slow_values.append(ema_slow)
    apo_values.append(ema_fast - ema_slow)

goog_data = goog_data.assign(FastExponential10DayMovingAverage=pd.Series(ema_fast_values, index=goog_data.index))
goog_data = goog_data.assign(SlowExponential40DayMovingAverage=pd.Series(ema_slow_values, index=goog_data.index))
goog_data = goog_data.assign(AbsolutePriceOscillator=pd.Series(apo_values, index=goog_data.index))

# chart for APO
fig = plt.figure()
ax1 = fig.add_subplot(211, ylabel='Google Price in $', title="APO")
goog_data.Close.plot(ax=ax1, color='g', lw=2, legend=True, label="Close Price")
goog_data.FastExponential10DayMovingAverage.plot(ax=ax1, color='r', lw=2., legend=True, label="Fast EMA 10 Days")
goog_data.SlowExponential40DayMovingAverage.plot(ax=ax1, color='b', lw=2., legend=True, label="Slow EMA 40 Days")
ax2 = fig.add_subplot(212, ylabel='APO')
goog_data.AbsolutePriceOscillator.plot(ax=ax2, color='black', lw=2., legend=True, label="APO")
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

# Moving Average Convergence Divergence --> Fast EMA - Slow EMA and then apply EMA MACD smoothing
time_period_fast = 10
time_period_slow = 40
time_period_macd = 20
K_fast = 2 / (time_period_fast + 1)
K_slow = 2 / (time_period_slow + 1)
K_macd = 2 / (time_period_macd + 1)
ema_fast = 0
ema_slow = 0
ema_macd = 0
ema_fast_values = []
ema_slow_values = []
macd_values = []  # store MACD values
macd_signal_values = []  # store MACD EMA values for smoothing MACD
macd_histogram_values = []  # MACD - MACD EMA

for close_price in goog_data.Close:
    if ema_fast == 0:
        ema_fast = close_price
        ema_slow = close_price
    else:
        ema_fast = close_price * K_fast + ema_fast * (1 - K_fast)
        ema_slow = close_price * K_slow + ema_slow * (1 - K_slow)
    ema_fast_values.append(ema_fast)
    ema_slow_values.append(ema_slow)
    macd = ema_fast - ema_slow

    if ema_macd == 0:
        ema_macd = macd
    else:
        ema_macd = macd * K_macd + ema_macd * (1 - K_macd)
    macd_values.append(macd)
    macd_signal_values.append(ema_macd)
    macd_histogram_values.append(macd - ema_macd)

goog_data = goog_data.assign(MovingAverageConvergenceDivergence=pd.Series(macd_values, index=goog_data.index))
goog_data = goog_data.assign(Exponential20DayMACD=pd.Series(macd_signal_values, index=goog_data.index))
goog_data = goog_data.assign(MACDHistogram=pd.Series(macd_histogram_values, index=goog_data.index))
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

# Chart for MACD
fig = plt.figure()
ax1 = fig.add_subplot(311, ylabel='Google Price in $', title="MACD")
goog_data.Close.plot(ax=ax1, color='g', lw=2, legend=True, label="Close Price")
goog_data.FastExponential10DayMovingAverage.plot(ax=ax1, color='r', lw=2., legend=True, label="Fast EMA 10 Days")
goog_data.SlowExponential40DayMovingAverage.plot(ax=ax1, color='b', lw=2., legend=True, label="Slow EMA 40 Days")
ax2 = fig.add_subplot(312, ylabel='MACD')
goog_data.MovingAverageConvergenceDivergence.plot(ax=ax2, color='g', lw=2., legend=True, label="MACD")
goog_data.Exponential20DayMACD.plot(ax=ax2, color='r', lw=2., legend=True, label="EMA MACD")

ax3 = fig.add_subplot(313, ylabel='MACD')
goog_data.MACDHistogram.plot(ax=ax3, color='black', kind='bar', legend=True, label="MACD", use_index=False)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

# Bollinger Band --> simple moving +/- (standard deviation factor * standard deviation)
# Higher the standard deviation factor, larger the width of the bands

time_period = 20
stdev_factor = 2
history = []
sma_values = []
upper_band = []
lower_band = []

for close_price in goog_data.Close:
    history.append(close_price)
    if len(history) > time_period:
        del (history[0])
    sma = stats.mean(history)
    sma_values.append(sma)

    variance = 0

    for hist_price in history:
        variance = variance + (hist_price - sma) ** 2
    stdev = math.sqrt(variance / len(history))
    upper_band.append(sma + stdev_factor * stdev)
    lower_band.append(sma - stdev_factor * stdev)

goog_data = goog_data.assign(UpperBollingerBand20Days=pd.Series(upper_band, index=goog_data.index))
goog_data = goog_data.assign(LowerBollingerBand20Days=pd.Series(lower_band, index=goog_data.index))

# chart for bollinger bands
fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Google Price in $', title="Bollinger Bands")
goog_data.Close.plot(ax=ax1, color='g', lw=2, legend=True, label="Close Price")
goog_data.Simple20DayMovingAverage.plot(ax=ax1, color='r', lw=2., legend=True, label="SMA")
goog_data.UpperBollingerBand20Days.plot(ax=ax1, color='b', lw=2., legend=True, label="Upper Band")
goog_data.LowerBollingerBand20Days.plot(ax=ax1, color='black', lw=2., legend=True, label="Lower Band")

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

# Relative Strength Indicator --> magnitude of average gains/loses over that period
time_period = 20
gain_history = []  # if price > previous price, gain = price - previous price. (0 if no gain)
loss_history = []  # if price < previous price, loss = previous price - price. (0 if no loss)
avg_gain_values = []  # for storing average gains
avg_loss_values = []  # for storing average losses
rsi_values = []
last_price = 0
# current price - last price > 0 then gain else loss

for close_price in goog_data.Close:
    if last_price == 0:
        last_price = close_price
    gain_history.append(max(0, close_price - last_price))
    loss_history.append(max(0, last_price - close_price))
    last_price = close_price

    if len(gain_history) > time_period:
        del gain_history[0]
        del loss_history[0]

    avg_gain = stats.mean(gain_history)
    avg_loss = stats.mean(loss_history)
    avg_gain_values.append(avg_gain)
    avg_loss_values.append(avg_loss)

    rs = 0
    if avg_loss > 0:
        rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values.append(rsi)

goog_data = goog_data.assign(RSIGain20Days=pd.Series(avg_gain_values, index=goog_data.index))
goog_data = goog_data.assign(RSILoss20Days=pd.Series(avg_loss_values, index=goog_data.index))
goog_data = goog_data.assign(RSI20Days=pd.Series(rsi_values, index=goog_data.index))

fig = plt.figure()
ax1 = fig.add_subplot(311, ylabel='Google Price in $', title="RSI")
goog_data.Close.plot(ax=ax1, color='g', lw=2, legend=True, label="Close Price")
ax2 = fig.add_subplot(312, ylabel='RS')
goog_data.RSIGain20Days.plot(ax=ax2, color='g', lw=2., legend=True, label="RSI Gain")
goog_data.RSILoss20Days.plot(ax=ax2, color='r', lw=2., legend=True, label="RSI Loss")
ax3 = fig.add_subplot(313, ylabel='RSI')
goog_data.RSI20Days.plot(ax=ax3, color='black', lw=2., legend=True, label="RSI")

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

# Momentum --> simply price(t) - price (t-n). Price(t-n) is price n time periods before time t
time_period = 20
history = []
mom_values = []  # store momentum values

for close_price in goog_data.Close:
    history.append(close_price)
    if len(history) > time_period:
        del (history[0])

    mom = close_price - history[0]
    mom_values.append(mom)

goog_data = goog_data.assign(MomentumFromPrice20Days=pd.Series(mom_values, index=goog_data.index))

# chart for momentum
fig = plt.figure()
ax1 = fig.add_subplot(211, ylabel='Google Price in $', title="Momentum")
goog_data.Close.plot(ax=ax1, color='g', lw=2, legend=True, label="Close Price")
ax2 = fig.add_subplot(212, ylabel='Standard Deviation')
goog_data.MomentumFromPrice20Days.plot(ax=ax2, color='black', lw=2., legend=True, label="Momentum 20 Days")
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()
print(goog_data)

df = pd.DataFrame(goog_data)
df.to_excel('Technical_Indicators.xlsx', index= False, header=True)
