import pandas as pd;
import numpy as np;
import datetime;
import yfinance as yf;

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader;
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split;
from finrl import config_tickers;
from finrl.config import INDICATORS;

import itertools;


#
# Fetch data with Yahoo Finance API
#

aapl_df_yf = yf.download(tickers='aapl', start='2020-01-01', end='2020-01-31');
aapl_df_yf.head();

print(aapl_df_yf.head());


#
# Fetch data with FinRL YahooDownloader
#

aapl_df_finrl = YahooDownloader(start_date='2020-01-01', end_date='2020-01-31', ticker_list=['aapl']).fetch_data();
aapl_df_finrl.head();



#
# Use Dow 30 symbols for analysis
#

dow30 = config_tickers.DOW_30_TICKER;


#
# Define Training and Trading start/end dates
#

TRAIN_START_DATE = '2009-01-01'; TRAIN_END_DATE = '2020-07-01';
TRADE_START_DATE = '2020-07-01'; TRADE_END_DATE = '2021-10-29';


#
# Fetch Dow 30 bar data
#

df_raw = YahooDownloader(start_date=TRAIN_START_DATE, end_date=TRADE_END_DATE, ticker_list=config_tickers.DOW_30_TICKER).fetch_data();



#
# Preprocess Data - Check for missing data, feature engineer data points into states
# 	• Add Technical Indicators (e.g. trend-following indicators MACD, RSI)
# 	• Add Turbulence Index (e.g. measure extreme flux in asset price)
#

fe = FeatureEngineer(
	use_technical_indicator = True,
	tech_indicator_list = INDICATORS,
	use_vix = True,
	use_turbulence = True,
	user_defined_feature = False
);

fe_processed = fe.preprocess_data(df_raw);

list_ticker = fe_processed['tic'].unique().tolist();
list_date = list(pd.date_range(fe_processed['date'].min(), fe_processed['date'].max()).astype(str));
combination = list(itertools.product(list_date, list_ticker));



#
# Process Data
#

processed_full = pd.DataFrame(combination, columns=['date', 'tic']).merge(fe_processed, on=['date', 'tic'], how='left');
processed_full = processed_full[processed_full['date'].isin(fe_processed['date'])];
processed_full = processed_full.sort_values(['date', 'tic']);
processed_full = processed_full.fillna(0);


























































































