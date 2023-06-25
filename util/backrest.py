import numpy as np
import pandas as pd
from binance.client import Client
from datetime import datetime
from datetime import timedelta
import statistics
import math
import os
from pathlib import Path
import talib.abstract as ta
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib.pyplot import style
from matplotlib.gridspec import GridSpec
from hurst import __get_simplified_RS
from hurst import __get_RS
from hurst import compute_Hc
#from hurst import __get_RS
from tqdm import tqdm
import warnings
import talib
import pandas_ta as ta
from util import resticator

## API Parameters
api_key = 'NejJl5zxuNPv4E3WRdhbohrUjAq9nyAZidahho7qrgRGlHi1cndnN2ajpLQGBlwT'
api_secret = 'aWyjfuE1QaovlARBAiif9DV8ehkvL7kxr12ig694K1J1rTM2rSJGNKn3FkytYsVR'
client = Client(api_key, api_secret)

global dd
global holding_period

def primary_data(sym, window, interval, download=True, override = False, time_from = None, time_to = None):
        
    date_from = time_from # format: "1 Dec, 2017"
    date_to = time_to
    
    if time_from != None:
        name = f'data/{sym}_{interval}_{window}_{str(time_from)}_{str(time_to)}.csv' ## change this to accomodate tf and tt
    else:
        name = f'data/{sym}_{interval}_{window}.csv' ## change this to accomodate tf and tt

    if download:
    #if not override: # override to download even if file exists
    #if not os.path.isfile(name):
        print("Downloading data..........")
        if interval == "1m":
            client_name = Client.KLINE_INTERVAL_1MINUTE
        elif interval == "3m":
            client_name = Client.KLINE_INTERVAL_3MINUTE
        elif interval == "5m":
            client_name = Client.KLINE_INTERVAL_5MINUTE
        elif interval == "15m":
            client_name = Client.KLINE_INTERVAL_15MINUTE
        elif interval == "30m":
            client_name = Client.KLINE_INTERVAL_30MINUTE
        elif interval == "1H":
            client_name = Client.KLINE_INTERVAL_1HOUR
        elif interval == "4H":
            client_name = Client.KLINE_INTERVAL_4HOUR
        elif interval == "1D":
            client_name = Client.KLINE_INTERVAL_1DAY

        if time_from != None:
            klines = client.get_historical_klines(sym, client_name, date_from, date_to)
        else:
            klines = client.get_historical_klines(sym, client_name, f"{window} days ago UTC")

        nl = []
        for data in klines:
            data_a = [j for i, j in enumerate(data) if i not in [6, 7, 8, 9, 10, 11]]
            data_a[0] = datetime.utcfromtimestamp(int(data[0])/1000)
            for i in range(5):
                data_a[i+1] = float(data_a[i+1])

            nl.append(data_a)

        del nl[-1]

        df_primary = pd.DataFrame(nl, columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']).set_index('Time')
        df = df_primary.copy()

        df.to_csv(name, date_format="%Y-%m-%d %H:%M:%S")

    else:
        df = pd.read_csv(name)
        df['Time'] = pd.to_datetime(df['Time'])
        df.set_index('Time', inplace=True)

    return df

def heikin_ashi(price_df):
    df = price_df.copy()
    df_HA = price_df.copy()
    df_HA['Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

    for i in range(0, len(df)):
        if i == 0:
            df_HA['Open'][i] = ( (df['Open'][i] + df['Close'][i] ) / 2)
        else:
            df_HA['Open'][i] = ( (df['Open'][i-1] + df['Close'][i-1] ) / 2)

    df_HA['High'] = df[['Open','Close','High']].max(axis=1)
    df_HA['Low'] = df[['Open','Close','Low']].min(axis=1)

    return df_HA

def fdi(data, lookback):
    hes = []
    for i in range(len(data)):
        if i > lookback:
            hc = 2 - compute_Hc(data[i - lookback:i])[0]
            hes.append(hc)
        else:
            hes.append(np.nan)

    return hes

def get_hurst_exponent(time_series, max_lag=20):
    """Returns the Hurst Exponent of the time series"""
    lags = range(2, max_lag)
    # variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)
    return reg[0]

def hurst(data, lookback):
    assert lookback>=100, "Lookback must be greater than 100"
    hes = []
    for i in range(len(data)):
        if i > lookback:
            hc = compute_Hc(data[i - lookback:i])[0]
            hes.append(hc)
        else:
            hes.append(np.nan)

    return hes

def rescaled_range(data, lookback, type):
    hes = []
    for i in range(len(data)):
        if i > lookback:
            if type=="normal":
                rr = __get_RS(data[i - lookback:i], "price")
            elif type == "simple":
                rr = __get_simplified_RS(data[i - lookback:i], "price")
            hes.append(rr)
        else:
            hes.append(np.nan)

    return hes

def cut_missing_data(mtf, HTF):
    if HTF == "4H":
        for _ in range(2):
            mtf.drop(mtf.index[-1], inplace=True)
        return mtf
    elif HTF == "1H":
        for _ in range(9):
            mtf.drop(mtf.index[-1], inplace=True)
        return mtf
    else:
        return mtf

def mtf2(ltf):
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    htf_close = []
    for i, row in ltf.iterrows():
        if i.day % 2 == 0 and i.hour == 23 and i.minute == 59 and i.second == 0:
            htf_close.append([i, row['Open'], row['High'], row['Low'], row['Close']])

    htf = pd.DataFrame(htf_close, columns=['Time', 'Open', 'High', 'Low', 'Close']).set_index('Time')
    ha = heikin_ashi(htf)

    ltf_close = pd.DataFrame({'Time':ltf.index.values,'Close':ltf['Close'].values})
    htf_close = pd.DataFrame({'Time':htf.index.values,'Close':htf['Close'].values})
    merged = pd.merge(ltf_close, htf_close, how='outer', on='Time')
    merged_complete = merged.fillna(method='ffill')

    return merged_complete

## No resampling - uses htf data
def mtf3(ltf, htf):
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    ltf_close = pd.DataFrame({'Time':ltf.index.values,'Close':ltf['Close'].values})
    htf_close = pd.DataFrame({'Time':htf.index.values,'Close':htf['Low'].values})
    merged = pd.merge(ltf_close, htf_close, how='outer', on='Time')
    merged_complete = merged.fillna(method='ffill')

    merged_complete['MTF'] = merged_complete.MTF.shift(24*60)
    print(merged_complete)

    return merged_complete

## Resampling
def mtf4(ltf, fill=False):
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    htf_close = []
    for i, row in ltf.iterrows():
        if i.day % 2 == 0 and i.hour == 23 and i.minute == 59 and i.second == 0:
            htf_close.append([i, row['Close']])

    htf = pd.DataFrame(htf_close, columns=['Time', 'Close']).set_index('Time')

    ltf_close = pd.DataFrame({'Time':ltf.index.values,'Close':ltf['Close'].values})
    htf_close = pd.DataFrame({'Time':htf.index.values,'Close_HTF':htf['Close'].values})
    merged = pd.merge(ltf_close, htf_close, how='outer', on='Time')
    if fill:
        merged_complete = merged.fillna(method='ffill')
    else:
        merged_complete = merged

    return merged_complete

## No resampling
def mtf5(ltf, htf, HTF, fill=False):
    if HTF == "4H":
        tf = 4
    elif HTF == "1D":
        tf = 24
    elif HTF == "1H":
        tf = 24
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    ltf_close = pd.DataFrame({'Time':ltf.index.values,'Close':ltf['Close'].values})
    htf_close = pd.DataFrame({'Time':htf.index.values,'MTF':htf['Close'].values})
    merged = pd.merge(ltf_close, htf_close, how='outer', on='Time')
    if fill:
        merged_complete = merged.fillna(method='ffill')
    else:
        merged_complete = merged

    merged_complete['MTF'] = merged_complete.MTF.shift(tf*60)
    df = cut_missing_data(merged_complete, HTF)
    #merged['MTF'] = merged.MTF.shift(tf*60)
    #print(merged)

    return df

def calmar(mdd, gain, window):
    days = window/(60*24)

    annual_ret = (1+gain)**(365/days) - 1
    calmar = annual_ret/mdd

    return calmar

def sortino(mar, pct_change_list):
    ## input:
    ## min acceptable return, returns (list)

    rf = 0.0000095129
    downside_dev = math.sqrt( sum(np.multiply(mar,mar)) / len(mar) )
    avg_return = statistics.mean(pct_change_list)
    stdev_return = statistics.stdev(pct_change_list)
    sr = (avg_return - rf) / downside_dev

    return sr

def sharpe(pct_change_list):
    avg_return = statistics.mean(pct_change_list)
    stdev_return = statistics.stdev(pct_change_list)
    rf = 0.0000095129
    sr = (avg_return - rf) / (stdev_return)

    return sr

def logic(ma_df, mtf_df, WINDOW, GRAPH, method):
    #Buy when SMA Crosses below FMA
    global CSIDE
    global SYMBOL
    global dd
    global holding_period

    long_list = []
    short_list = []
    intra_trade_price = []
    #intra_trade_prices = []
    dd = []
    holding_period = []
    llnn = []
    slnn = []
    positions = []
    is_crossed = False
    for i, row in ma_df.iterrows():
        if method == "crossdown":
            if row['ema'] < row['MTF'] and not is_crossed:
                ## Cross down
                ## Go long
                is_crossed = True
                CSIDE = 1
                cross = [i, row['Close']]
                long_list.append(cross)
                llnn.append(row['Close'])
            elif row['ema'] > row['MTF'] and is_crossed:
                ## Cross up
                ## Go short
                is_crossed = False
                CSIDE = 0
                cross = [i, row['Close']]
                slnn.append(row['Close'])
                short_list.append(cross)
                #intra_trade_prices.append(intra_trade_price)
                if len(intra_trade_price) > 0 and len(llnn) > 0:
                    dd.append( (llnn[-1] - min(intra_trade_price)) / min(intra_trade_price) )
                holding_period.append(len(intra_trade_price))
                intra_trade_price = []
                if len(llnn) != 0:
                    positions.append([i, llnn[-1], row['Close']])
            else:
                cross = [i, np.nan]
                short_list.append(cross)
                long_list.append(cross)

            if is_crossed:
                intra_trade_price.append(row['Close'])
        elif method == "crossup":
            if row['ema'] > row['MTF'] and not is_crossed:
                ## Cross Up
                ## Go long
                is_crossed = True
                CSIDE = 1
                cross = [i, row['Close']]
                long_list.append(cross)
                llnn.append(row['Close'])
            elif row['ema'] < row['MTF'] and is_crossed:
                ## Cross down
                ## Go short
                is_crossed = False
                CSIDE = 0
                cross = [i, row['Close']]
                slnn.append(row['Close'])
                short_list.append(cross)
                #intra_trade_prices.append(intra_trade_price)
                if len(intra_trade_price) > 0 and len(llnn) > 0:
                    dd.append( (llnn[-1] - min(intra_trade_price)) / min(intra_trade_price) )
                holding_period.append(len(intra_trade_price))
                intra_trade_price = []
                if len(llnn) != 0:
                    positions.append([i, llnn[-1], row['Close']])
            else:
                cross = [i, np.nan]
                short_list.append(cross)
                long_list.append(cross)

            if is_crossed:
                intra_trade_price.append(row['Close'])


    long_df = pd.DataFrame(long_list, columns = ['opentime', 'long']).set_index('opentime')
    short_df = pd.DataFrame(short_list, columns = ['opentime', 'short']).set_index('opentime')
    pos_df = pd.DataFrame(positions, columns = ['opentime', 'buy', 'sell']).set_index('opentime')

    ## PLOTTING ##
    if GRAPH:
        mtf_df['Close'].plot()
        mtf_df['MTF'].plot()
        mtf_df['ema'].plot()

        plt.scatter(long_df.index.values, long_df['long'].values, marker = '^', color='green', s=200)
        plt.scatter(short_df.index.values, short_df['short'].values, marker = 'v', color='red', s=200)
        plt.legend(["Close", "HTF", "EMA", "Long", "Short"])
        plt.show()

    return long_df, short_df, pos_df

def metrics(pos_df, WINDOW, SYMBOL, PRINTSTATS, GRAPH, init_acct_val = 1000):
    ###########################################################################
    ## Metrics    #############################################################
    ###########################################################################
    #global dd
    change_list = []
    cum_change_list = []
    pct_change_list = []
    mar = []
    acct_val = init_acct_val
    equity_curve = [acct_val]
    wins = 0
    losses = 0
    profit = 0
    loss = 0
    for i, row in pos_df.iterrows():
        change = (row['Sell'] - row['Buy'])
        cum_change = acct_val*change
        pct_change = (row['Sell'] - row['Buy'])/row['Buy'] - 0.075/50
        pct_downside = pct_change - (0.3/100)
        change_list.append(change)
        cum_change_list.append(cum_change)
        pct_change_list.append(pct_change)
        mar.append(pct_downside)
        acct_val = acct_val*(1+pct_change)
        equity_curve.append(acct_val)

    loss_list = [change for change in change_list if change < 0]
    losses = len(loss_list)
    loss = sum(loss_list)
    profit_list = [change for change in change_list if change > 0]
    wins = len(profit_list)
    profit = sum(profit_list)

    if (wins+losses) == 0:
        print("No trades made")
    else:
        accuracy = round( (wins/(losses+wins)) * 100 , 2)

    for i, j in enumerate(mar):
        if j > 0:
            mar[i] = 0
        elif j < 0:
            continue

    shrpe = round(sharpe(pct_change_list),2)
    srtno = round(sortino(mar, pct_change_list),2)
    profit_factor = round(sum(profit_list) / abs(sum(loss_list)), 2)
    #dd = metrics.max_drawdown(equity_curve, WINDOW)
    mdd = 1 #round(max(dd)*100, 2)
    #calmar = metrics.calmar(mdd, equity_curve, WINDOW)
    ## Holding period in minutes
    maxhold = 1 #round(int(max(holding_period)) / (24*60), 2)# in days
    minhold = 1 #round(int(min(holding_period)) / (24*60),4) # in days
    avghold = 1 #round(statistics.mean(holding_period) / (24*60), 2)
    goa = round(((acct_val - init_acct_val) / init_acct_val) * 100, 2)
    ntrades = len(pos_df.index)
    avgtrade = round(statistics.mean(pct_change_list)*100,2)
    hgain = round(max(pct_change_list)*100, 3)
    hloss = round(min(pct_change_list)*100, 3)

    if PRINTSTATS:
        print(max(pct_change_list)*100)
        print("---------------------------------------------------------------------------------------------")
        print(f"Accuracy = {accuracy}%")
        print(f"Net profit is ${profit} and net loss is ${loss}")
        print(f"Final account value is ${acct_val}")
        print(f"Gain on account = {goa} %")
        print(f"Sharpe Ratio is {shrpe}")
        print(f"Sortino is {srtno}")
        print(f"Profit factor is {profit_factor}")
        print(f"Number of trades is: {ntrades} in {WINDOW} days")
        print(f"Avg trade = {avgtrade}%")
        print(f"Highest gain is: {hgain}%")
        print(f"Highest loss is: {hloss}%")
        print(f"Max drawdown is {mdd}%")
        print(f"Max holding period is: {maxhold} days")
        print(f"Min holding period is: {minhold} days")
        print(f"Avg holding period is: {avghold} days")
        print("---------------------------------------------------------------------------------------------")

    if GRAPH:
        stats1 = [[str(accuracy) + "%"], ["$" + str(profit)], ["$"+str(loss)], ["$" + str(acct_val)], [str(goa) + "%"], [shrpe], [srtno], [profit_factor]]
        stats2 = [[ntrades], [str(avgtrade) + "%"], [str(hgain) + "%"], [str(hloss) + "%"], [str(mdd) + "%"], [str(maxhold) + " days"], [str(minhold) + " days"],\
        [str(avghold) + " days"]]

        stats_df1 = pd.DataFrame(stats1)
        stats_df2 = pd.DataFrame(stats2)

        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(3, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :])
        ax3 = fig.add_subplot(gs[-1, 0])
        ax4 = fig.add_subplot(gs[-1, -1])

        ax1.plot(np.arange(0,len(equity_curve)), equity_curve, color="mediumvioletred")
        #ax2.plot(np.arange(0,len(dd)), np.multiply(dd,-100), color="mediumvioletred")
        ax1.grid(True)
        ax2.grid(True)
        ax3.axis('tight')
        ax3.axis('off')
        ax3.table(cellText=stats_df1.values, rowLabels=["Accuracy","Net profit", "Net loss", "Account Val", "GOA", "Sharpe", "Sortino", "Profit Factor"],loc="center")
        ax4.axis('tight')
        ax4.axis('off')
        ax4.table(cellText=stats_df2.values, rowLabels=["Number of trades","Avg trade return", "Highest gain", "Highest loss", "Max DD", "Max hold period", "Min hold period", "Avg hold period"],loc="center")

        ax1.set_title(f"Equity Curve for {SYMBOL} over {WINDOW} days.", weight='bold')
        ax2.set_title("Drawdown", weight='bold')
        ax2.set_xlabel("Trade number", weight='bold')
        ax1.set_ylabel("Account value ($)", weight='bold')
        ax2.set_ylabel("Drawdown (%)", weight='bold')
        plt.show()

    return accuracy, shrpe, srtno

def regress(ltf, power, j, column_name):
    ## ltf = DataFrame column with time index
    ## j = regression window
    ## Returns list
    gradients = []
    tip = []
    for i in tqdm(range(len(ltf))):
        if i > j:
            df = ltf[i - j: i]
            current_time = df.index.values[-1]

            ## LINREG
            x = np.arange(0, len(df))
            y = df[column_name].values
            preg = np.poly1d(np.polyfit(x, y, power))
            pline = np.linspace(0, len(df), 100)

            linregline = preg(pline)
            gradient = (linregline[-1] - linregline[0]) / j
            gradients.append([current_time, gradient])
            tip.append([current_time, linregline[-1]])

            ## EXTRA FUNCTIONALITY - IGNORE
            #if gradient > 0 and obvgradient < 0:
            #    #print("here1")
            #    div.append([current_time, 0])
            #elif gradient < 0 and obvgradient > 0:
            #    #print("here2")
            #    div.append([current_time, 0])
            #else:
            #    div.append([current_time, np.nan])

        else:
            gradients.append([ltf.reset_index().loc[i, "Time"], np.nan])
            tip.append([ltf.reset_index().loc[i, "Time"], np.nan])

    return tip, gradients

def crossover(x, y, i):
    # x = series
    # y = series
    is_crossed = False
    if x[i] > y[i] and not is_crossed:
        is_crossed = True
        return True
    elif x[i] < y[i] and is_crossed:
        is_crossed = False
        return True
    else:
        return False

def save(df, dirname, filename):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    i = 0
    while os.path.exists(f"data/{dirname}/{filename}_%s.csv.xml" % i):
        i += 1

    #fh = open("sample%s.xml" % i, "w")

    if os.path.isfile(filename):
        df.to_csv(f"data/{dirname}/{filename}.csv")
    else:
        df.to_csv(f"data/{dirname}/{filename}_1.csv")

# preallocate empty array and assign slice by chrisaycock
def shift_array(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def fftPlot(sig, dt=None, plot=True):
    # Here it's assumes analytic signal (real signal...) - so only half of the axis is required

    if dt is None:
        dt = 1
        t = np.arange(0, sig.shape[-1])
        xLabel = 'samples'
    else:
        t = np.arange(0, sig.shape[-1]) * dt
        xLabel = 'freq [Hz]'

    if sig.shape[0] % 2 != 0:
        warnings.warn("signal preferred to be even in size, autoFixing it...")
        t = t[0:-1]
        sig = sig[0:-1]

    sigFFT = np.fft.fft(sig) / t.shape[0]  # Divided by size t for coherent magnitude

    freq = np.fft.fftfreq(t.shape[0], d=dt)

    # Plot analytic signal - right half of frequence axis needed only...
    firstNegInd = np.argmax(freq < 0)
    freqAxisPos = freq[0:firstNegInd]
    sigFFTPos = 2 * sigFFT[0:firstNegInd]  # *2 because of magnitude of analytic signal

    if plot:
        plt.figure()
        plt.plot(freqAxisPos, np.abs(sigFFTPos))
        plt.xlabel(xLabel)
        plt.ylabel('mag')
        plt.title('Analytic FFT plot')
        plt.show()

    return sigFFTPos, freqAxisPos

def redline(highs, lows, closes, lookback):
    ## Take np arrays
    is_crossed = False
    PP = []
    S3 = []
    line = []
    line_unique = []
    l_l = 0
    for i in range(len(closes)):
        if i > 2:
            ph = highs[i-1]
            pl = lows[i-1]
            pc = closes[i-1]
            PP.append((ph+pl+pc)/3)
            if len(PP) > 0:
                S3.append(pl - lookback * (ph-PP[i])) #5
            else:
                S3.append(np.nan)

            if closes[i] > S3[i] and not is_crossed:
                is_crossed = True
                l = True
            elif closes[i] < S3[i] and is_crossed:
                is_crossed = False
                #l = True
            else:
                l = False

            if l:
                line_unique.append([i, l_l])
                l_l = closes[i]
                l = False
            else:
                if len(line) > 0:
                    l_l = line[-1]
                else:
                    l_l = np.nan
                l = False

            line.append(l_l)
        else:
            line.append(np.nan)
            PP.append(np.nan)
            S3.append(np.nan)

    return line, line_unique

def get_symbols():
    client = Client(api_key, api_secret)
    exchange_info = client.get_exchange_info()
    #symbol_list = sorted([s['symbol'] for s in exchange_info['symbols'] if s['symbol'][-4:] == 'USDT' and "up" not in s['symbol'] and "down" not in s['symbol']])
    symbol_list = []
    for s in exchange_info['symbols']:
        if s['symbol'][-4:] == 'USDT' and "DOWN" not in s['symbol'] and "UP" not in s['symbol'] and s['symbol'] != "BCCUSDT" and s['symbol'] != "BCHABCUSDT" and s['symbol'] != "BCHSVUSDT":
            symbol_list.append(s['symbol'])

    return sorted(symbol_list)

def vosc(data):
    short_ema = talib.EMA(data['Volume'].values, 5)
    long_ema = talib.EMA(data['Volume'].values, 10)
    vol_osc = 100*(short_ema - long_ema) / long_ema
    
    return vol_osc

def get_indicators(raw_data):
    data = raw_data.copy(deep=True)
    # overlap indicators (need to add ichimoku)
    overlap_ind = "alma, dema, ema_25, ema_9, fwma, hilo, hl2, hlc3, hma, hwma, kama, linreg, mcgd, midpoint, midprice, ohlc4, pwma, rma, sinwma, sma, ssf, supertrend, swma, t3, tema, trima, vidya, vwap, vwma, wcp, wma, zlma"
    overlap_ind = overlap_ind.split(", ")

    for indi in overlap_ind:
        try:
            if indi[:3] == "ema":
                ema_period = int(''.join(filter(str.isdigit, indi)))
                indicator_data = raw_data.copy().ta.ema(ema_period)
            else:
                indicator_data = getattr(raw_data.copy().ta, indi)()
            indicator_data = raw_data["Close"]/indicator_data - 1
            if isinstance(indicator_data, pd.DataFrame):
                data.join(indicator_data)
            elif isinstance(indicator_data, pd.Series):
                data[indi] = indicator_data
            else:
                pass
        except Exception as e:
            print(f"Exception occurred on {indi}:", e)
            pass
    
    # ichimoku - keeps giving errors :(
    #ichimoku = raw_data.copy().ta.ichimoku(lookahead=False)[0]
    #ichimoku_spanA = ichimoku["ISA_9"].values
    #data["ichimoku_spanA_dev"] = raw_data["Close"].values/ichimoku_spanA - 1

    #ichimoku_spanB = ichimoku["ISB_26"].values
    #data["ichimoku_spanB_dev"] = raw_data["Close"].values/ichimoku_spanB - 1
    
    # momentum indicators
    mom_ind = "ao, apo, bias, bop, brar, cci, cfo, cg, cmo, coppock, cti, dm, er, eri, fisher, inertia, kdj, kst, macd, mom, pgo, ppo, psl, pvo, qqe, roc, rsi, rsx, rvgi, stc, slope, smi, squeeze, squeeze_pro, stoch, stochrsi, td_seq, trix, tsi, uo, willr, ebsw"
    mom_ind = mom_ind.split(", ")

    for indi in mom_ind:
        try:
            indicator_data = getattr(raw_data.copy().ta, indi)()

            if indi == "coppock":
                indicator_data = indicator_data.pct_change()
                #indicator_data = np.log(indicator_data) - np.log(indicator_data.shift(1))
            else:
                pass

            if isinstance(indicator_data, pd.DataFrame):
                data.join(indicator_data)
            elif isinstance(indicator_data, pd.Series):
                data[indi] = indicator_data
            else:
                pass

        except Exception as e:
            print(f"Exception occurred on {indi}:", e)
            #print(indicator_data.isna().any())
            #print(indicator_data.to_string())
            pass
        
    data["vosc"] = vosc(raw_data)
    data["lrsi"] = resticator.LRSI(data, 14)
    
    # statistical indicators
    stat_ind = "entropy, kurtosis, mad, median, quantile, skew, stdev, tos_stdevall, variance, zscore"
    stat_ind = stat_ind.split(", ")

    for indi in stat_ind:
        try:
            indicator_data = getattr(raw_data.copy().ta, indi)()
            if isinstance(indicator_data, pd.DataFrame):
                data.join(indicator_data)
            elif isinstance(indicator_data, pd.Series):
                data[indi] = indicator_data
            else:
                pass

        except Exception as e:
            print(f"Exception occurred on {indi}:", e)
            pass
    
    # trend indicators
    trend_ind = "adx, amat, aroon, chop, cksp, decay, decreasing, dpo, increasing, long_run, psar, qstick, short_run, tsignals, ttm_trend, vhf, vortex, xsignals"
    trend_ind = trend_ind.split(", ")

    for indi in trend_ind:
        try:        
            if indi == "dpo":
                indicator_data = raw_data.copy().ta.dpo(lookahead=False, centered=False)
            else:
                indicator_data = getattr(raw_data.copy().ta, indi)()

            if isinstance(indicator_data, pd.DataFrame):
                data.join(indicator_data)
            elif isinstance(indicator_data, pd.Series):
                data[indi] = indicator_data
            else:
                pass

        except Exception as e:
            print(f"Exception occurred on {indi}:", e)
            #print(indicator_data.to_string())
            pass
    
    # volume indicators
    vol_ind = "ad, adosc, aobv, cmf, efi, kvo, mfi, nvi, obv, pvi, pvol, pvr, pvt"
    vol_ind = vol_ind.split(", ")

    for indi in vol_ind:
        try:
            indicator_data = getattr(raw_data.copy().ta, indi)()

            if indi == "aobv" or indi == "obv":
                indicator_data = indicator_data.pct_change()
                #indicator_data = np.log(indicator_data) - np.log(indicator_data.shift(1))
            elif indi == "ad":
                data["ad_grad"] = np.gradient(indicator_data)
                
            if isinstance(indicator_data, pd.DataFrame):
                data.join(indicator_data)
            elif isinstance(indicator_data, pd.Series):
                data[indi] = indicator_data
            else:
                pass

        except Exception as e:
            print(f"Exception occurred on {indi}:", e)
            #print(indicator_data)
    
    # volatility indicators
    volatility_ind = "aberration, accbands, atr, bbands, donchian, hwc, kc, massi, natr, pdist, rvi, thermo, true_range, ui"
    volatility_ind = volatility_ind.split(", ")
    for indi in volatility_ind:
        try:
            indicator_data = getattr(raw_data.copy().ta, indi)()

            if indi == "aberration" or indi == "hwc" or indi == "bbands" or indi == "donchian" or indi == "kc" \
            or indi == "accbands":
                indicator_data = raw_data["Close"]/indicator_data - 1

            if indi == "atr" or indi == "natr" or indi == "thermo" or indi == "pdist" or indi == "true_range":
                indicator_data = indicator_data.pct_change()
                #indicator_data = np.log(indicator_data) - np.log(indicator_data.shift(1))

            if isinstance(indicator_data, pd.DataFrame):
                data.join(indicator_data)
            elif isinstance(indicator_data, pd.Series):
                data[indi] = indicator_data
            else:
                pass

        except Exception as e:
            print(f"Exception occurred on {indi}:", e)
            #print(indicator_data)
    
    # other
    ## Hurst Exponent - numerical (or categorical?)
    '''try:
        hurst_exp, hurst_type = resticator.HURST_EXP(data, 100)
        data["hurst_exp"] = hurst_exp
        data["hurst_type"] = hurst_type
    except FloatingPointError as e:
        print("Exception :(")
        print(e)'''
        
    hurst_exp = resticator.HURST_EXP(data, 20)
    hurst_type = 0 if 0 < hurst_exp < 0.5 else 1
    
    # price action
    n_green_candles = 0
    n_red_candles = 0
    for i, row in raw_data.iterrows():
        if row["Close"] >= row["Open"]:
            n_green_candles+=1
        else:
            n_red_candles+=1

    data = data.drop(["Open", "High", "Low", "Close", "Volume"], axis=1)
    #data_last_row = data.iloc[-1].values
    #data_last_row = np.append(data_last_row, [hurst_exp, hurst_type])
    #data_last_row = np.append(data_last_row, n_green_candles)
    #data_last_row = np.append(data_last_row, n_red_candles)
    #df_columns = np.append(data.columns.values, ["hurst_exp", "hurst_type"])
            
    return data