# TradingModel
## Introduction
This model takes a momentum-based trading strategy and filters signals into "buy" and "sell" based on an XGBoost classification algorithm. Specifically, the algorithm forms what I call a "Candle Pocket" for the signal by taking a window of the OHLCV time-series data prior to the signal generated by the momentum-based strategy. It then uses the Candle Pocket time-series data to generate a large collection of features for the machine learning algorithm, namely time-series characteristics such as entropy and autocorrelation, as well as traditional market indicators such as RSI and stochastic oscillators. Relevant features are then extracted based on statistical significance tests, and the newly formed Candle Pockets are used to train an XGBoost model, classifying each Candle Pocket as a "buy" or a "sell".

## Files
### SymbolPreprocess
This file downloads 