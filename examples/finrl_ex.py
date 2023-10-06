import pandas as pd;
import numpy as np;
import datetime;
import itertools;
import yfinance as yf;

# Fetching/Processing Data
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader;
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split;
from finrl import config_tickers;
from finrl.config import INDICATORS;


# Training Model/Agent
from stable_baselines3.common.logger import configure;
from finrl.agents.stablebaselines3.models import DRLAgent;
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR;
from finrl.main import check_and_make_directories;
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv;

check_and_make_directories([TRAINED_MODEL_DIR]);




#
# BEGIN PART ONE - FETCHING/PROCESSING DATA
#





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



#
# Partition data set into Training/Trading sets
#

train_df = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE);
trade_df = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE);



#
# BEGIN TUTORIAL NUMBER 2 - TRAINING THE TRADING AGENT
#

train = train_df;
# train = train_df.set_index(train_df.columns[0]);
# train.index.names = [''];



#
# Construct market environment
#

stock_dimension = len(train['tic'].unique());
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension;

print('Stock Dimension: {}\t State Space: {}'.format(stock_dimension, state_space));


buy_cost_list = [.001] * stock_dimension;
sell_cost_list = [.001] * stock_dimension;

num_stock_shares = [0] * stock_dimension;

env_kwargs = {
	'hmax': 100,
	'initial_amount': 1000000,
	'num_stock_shares': num_stock_shares,
	'buy_cost_pct': buy_cost_list,
	'sell_cost_pct': sell_cost_list,
	'state_space': state_space,
	'stock_dim': stock_dimension,
	'tech_indicator_list': INDICATORS,
	'action_space': stock_dimension,
	'reward_scaling': 1e-4
}


e_train_gym = StockTradingEnv(df=train, **env_kwargs);


#
# Create Trading Environment
#

env_train, _ = e_train_gym.get_sb_env(); print(type(env_train));





#
# Train Deep Reinforcement Learning Agents
#

agent = DRLAgent(env=env_train);


#
# Define algorithms to employ (set to True to use)
#

if_using_a2c 	= True; # Advantage Actor Critic
if_using_ddpg 	= True; # Deep Deterministic Policy Gradient
if_using_ppo 	= True; # Proximal Policy Optimization
if_using_td3 	= True; # Twin-Delayed Deep Deterministic Policy Gradient
if_using_sac 	= True; # Soft-Agent Critic



#
# Train Agent on Five Algorithms - A2C, DDPG, PPO, TD3, SAC
#


# Agent 1 - A2C (Advantage Actor Critic)
agent = DRLAgent(env=env_train);
model_a2c = agent.get_model('a2c');

if if_using_a2c:
	tmp_path = RESULTS_DIR + '/a2c'; # logger
	new_logger_a2c = configure(tmp_path, ['stdout', 'csv', 'tensorboard']);
	model_a2c.set_logger(new_logger_a2c); # new logger

trained_a2c = agent.train_model(model=model_a2c, tb_log_name='a2c', total_timesteps=30000) if if_using_a2c else None;
trained_a2c.save(TRAINED_MODEL_DIR + '/agent_a2c') if if_using_a2c else None;


# Agent 2 - DDPG (Deep Deterministic Policy Gradient)
agent = DRLAgent(env=env_train);
model_ddpg = agent.get_model('ddpg');

if if_using_ddpg:
	tmp_path = RESULTS_DIR + '/ddpg';
	new_logger_ddpg = configure(tmp_path, ['stdout', 'csv', 'tensorboard']);
	model_ddpg.set_logger(new_logger_ddpg);

trained_ddpg = agent.train_model(model=model_ddpg, tb_log_name='ddpg', total_timesteps=30000) if if_using_ddpg else None;
trained_ddpg.save(TRAINED_MODEL_DIR + '/agent_ddpg') if if_using_ddpg else None;


# Agent 3 - PPO (Proximal Policy Optimization)
agent = DRLAgent(env=env_train);

PPO_PARAMS = {
	'n_steps': 2048,
	'ent_coef': .01,
	'learning_rate': 2.5e-4,
	'batch_size': 128
};

model_ppo = agent.get_model('ppo', model_kwargs=PPO_PARAMS);

if if_using_ppo:
	tmp_path = RESULTS_DIR + '/ppo';
	new_logger_ppo = configure(tmp_path, ['stdout', 'csv', 'tensorboard']);
	model_ppo.set_logger(new_logger_ppo);

	trained_ppo = agent.train_model(model=model_ppo, tb_log_name='ppo', total_timesteps=200000);
	trained_ppo.save(TRAINED_MODEL_DIR + '/agent_ppo');


# Agent 4 - TD3 (Twin-Delayed Deep Deterministic Policy Gradient)
agent = DRLAgent(env=env_train);
TD3_PARAMS = {'batch_size':100, 'buffer_size':1000000, 'learning_rate':.001};

model_td3 = agent.get_model('td3', model_kwargs = TD3_PARAMS);

if if_using_td3:
	model_td3.set_logger(configure(RESULTS_DIR + '/td3', ['stdout', 'csv', 'tensorboard']));
	trained_td3 = agent.train_model(model=model_td3, tb_log_name='td3', total_timesteps=50000);
	trained_td3.save(TRAINED_MODEL_DIR + '/agent_td3');


# Agent 5 - SAC (Soft-Actor Critic)
agent = DRLAgent(env=env_train);

SAC_PARAMS = {
	'batch_size': 128,
	'buffer_size': 100000,
	'learning_rate': .0001,
	'learning_starts': 100,
	'ent_coef': 'auto_0.1'
};

model_sac = agent.get_model('sac', model_kwargs=SAC_PARAMS);

if if_using_sac:
	model_sac.set_logger(configure(RESULTS_DIR + '/sac', ['stdout', 'csv', 'tensorboard']));
	trained_sac = agent.train_model(model=model_sac, tb_log_name='sac', total_timesteps=70000);
	trained_sac.save(TRAINED_MODEL_DIR + '/agent_sac');




































































