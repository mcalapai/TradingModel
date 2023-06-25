import numpy as np
import pandas as pd
import talib
import pandas_ta as ta
from pandas_ta.utils import get_offset, verify_series
from hurst import compute_Hc

def rma(close, length=None, offset=None, **kwargs):
    """Indicator: wildeR's Moving Average (RMA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    alpha = (1.0 / length) if length > 0 else 0.5
    close = verify_series(close, length)
    offset = get_offset(offset)
    
    if close is None: 
        print("here")
        return

    # Calculate Result
    rma = close.ewm(alpha=alpha, min_periods=length).mean()

    # Offset
    if offset != 0:
        rma = rma.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        rma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        rma.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    rma.name = f"RMA_{length}"
    rma.category = "overlap"

    return rma

def LRSI(data, length):
    raw_data = data.copy()
    raw_data.reset_index(inplace=True)
    
    a = raw_data['Close'].rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0])
    b = raw_data['Volume'].rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0])
    
    p1 = []
    for i, j in zip(a, b):
        p1.append([max(i, 0) * max(j, 0), abs(i) * abs(j)])
        
    params = pd.DataFrame(p1, columns=["p1", "p2"])
    lrsi = rma(params["p1"], length=length) / rma(params["p2"], length = length)
    
    return lrsi

from hurst import compute_Hc

def HURST(data, lookback):
    raw_data = data.copy()
    assert lookback >= 100, "Lookback must be greater than or equal to 100!"
    hes = []
    for i in range(len(raw_data)):
        if i > lookback:
            hc_window = raw_data["Close"][i - lookback:i]
            hc = compute_Hc(hc_window)[0]
            hes.append(hc)
        else:
            hes.append(np.nan)

    raw_data["hurst"] = hes
    behavior_type = raw_data["hurst"].apply(lambda x: 0 if 0 < x < 0.5 else 1)
    
    return raw_data["hurst"].values, behavior_type.values

def get_hurst_exponent(time_series, max_lag=20):
    """Returns the Hurst Exponent of the time series"""
    lags = range(2, max_lag)
    # variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)
    return reg[0]

def HURST_EXP(data, lag):
    raw_data = data.copy()
    hurst_exp = get_hurst_exponent(data["Close"].values, lag)
    
    return hurst_exp

#####################################################################################
## Support/Resistance Levels ########################################################
#####################################################################################

def detect_level_method_2(df):
    levels = []
    max_list = []
    min_list = []
    for i in range(5, len(df)-5):
        high_range = df['High'][i-5:i+4]
        current_max = high_range.max()
        if current_max not in max_list:
            max_list = []
        max_list.append(current_max)
        if len(max_list) == 5 and is_far_from_level(current_max, levels, df):
            levels.append((high_range.idxmax(), current_max))
      
        low_range = df['Low'][i-5:i+5]
        current_min = low_range.min()
        if current_min not in min_list:
            min_list = []
        min_list.append(current_min)
        if len(min_list) == 5 and is_far_from_level(current_min, levels, df):
            levels.append((low_range.idxmin(), current_min))
    return levels