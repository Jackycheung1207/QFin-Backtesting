from pprint import pprint
from typing import Optional

import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import plotly.figure_factory as ff
from datetime import datetime
import plotly.graph_objects as go
from loguru import logger
from decimal import Decimal, ROUND_HALF_UP
import itertools
from util.util_performance import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import plotly.io as pio
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

class Backtester:
    def __init__(self,
                 metric_type: str = 'price',metric_name: str = 'BTC' , coin: str ='BTC', tf: str = '1d',
                 is_momentum: bool = True, threshold: float = None, lookback_period: int= None,
                 denoise_logic: str = 'z', shift: int = 0, style:str = 'long&short' ,training_size: float=0.7,
                 exchange:str = 'binance', instrument: str = 'perp', unit_tc: float = 0.0006):
        self.metric_type = metric_type
        self.metric_name = metric_name
        self.coin = coin
        self.tf = tf
        self.tf_int = int(tf.replace('h', ''))
        self.is_momentum = is_momentum
        self.threshold = threshold
        self.lookback_period = lookback_period
        self.denoise_logic = denoise_logic
        self.shift = shift
        self.style = style
        self.training_size = training_size
        self.exchange = exchange
        self.instrument = instrument
        self.unit_tc = unit_tc
        
        
        self.START_DATE = '2020-06-01'
        self.END_DATE = '2024-04-30'
        self.PRICE_FOLDER_PATH = '../Data/Price'
        self.METRIC_FOLDER_PATH = f'../Data/OnChain'
        self.RESULT_FOLDER_PATH = f'result/{self.metric_name}'


    def assign_position(self,df:pd.DataFrame, threshold: float) -> pd.DataFrame:

        """
        Assign positions to a dataframe based on the signal column.

        Args:
            df: The dataframe to assign positions to.
            threshold: The threshold value.

        Returns:
            The dataframe with the position column added.
        """
        if threshold is not None:
            if not self.is_momentum:
                # Assign positions based on the signal column
                df['position'] = np.where(df['signal'] > threshold, -1,
                                          np.where(df['signal'] < -threshold, 1, 0))
            else:
                # Assign positions based on the signal column
                df['position'] = np.where(df['signal'] > threshold, 1,
                                          np.where(df['signal'] < -threshold, -1, 0))
        else:
            if not self.is_momentum:
                # Assign positions based on the signal column
                df['position'] = np.where(df['signal'] > 0, -1,
                                          np.where(df['signal'] < 0, 1, 0))
            else:
                # Assign positions based on the signal column
                df['position'] = np.where(df['signal'] > 0, 1,
                                          np.where(df['signal'] < 0, -1, 0))

        return df

    def denoise(self, df: pd.DataFrame, lookback_period: int) -> pd.DataFrame:
        if self.denoise_logic == 'z':
            df['ma'] = df['metric'].rolling(lookback_period).mean()
            df['sd'] = df['metric'].rolling(lookback_period).std()
            df['signal'] = (df['metric'] - df['ma']) / df['sd']

        elif self.denoise_logic == 'pct':
            df['signal'] = df['metric'].pct_change()

        elif self.denoise_logic == 'ma_diff':
            df['ma'] = df['metric'].rolling(lookback_period).mean()
            df['signal'] = (df['metric'] / df['ma']) - 1

        return df

    def compute_expected_return_(self,df: pd.DataFrame):
        if self.style == 'long&short':
            suffix = ''
        elif self.style == 'long_only':
            suffix = '_long_only'
        elif self.style == 'short_only':
            suffix = '_short_only'

        df['trades'] = df['position' + suffix] != df['position' + suffix].shift(1)

        df = df[df.trades]
        trades = len(df)
        df['p'] = df['cum_strategy_ret' + suffix] - df['cum_strategy_ret' + suffix].shift(1)
        df['wl'] = df.p > 0  # bool

        win_rate = (df.wl.value_counts().tolist()[0] / trades) * 100
        win_rate = round(win_rate, 2)
        lose_rate = 1 - win_rate
        lose_rate = round(lose_rate, 2)

        win_pnl = df[df.wl].p.mean() * 100
        lose_pnl = df[df.wl < 1].p.mean() * 100
        win_pnl = round(win_pnl, 2)
        lose_pnl = round(lose_pnl, 2)

        expected_return = round((win_rate * win_pnl + lose_rate * lose_pnl) / 100, 2)

        return expected_return, win_rate, win_pnl, lose_pnl

    def get_price_df(self) -> pd.DataFrame:
        price_file_name = f'{self.coin}_{self.tf}_{self.exchange}_{self.instrument}.csv'
        logger.info(f"READ: {price_file_name}")
        price_df = pd.read_csv(
            os.path.join(self.PRICE_FOLDER_PATH, price_file_name)
        )[['timestamp', 'close']]
        # price_df = price_df.rename(columns={'close': 'price'})
        self.price_df = price_df
        return price_df

    def get_price_metric_df(self) -> pd.DataFrame:
        price_file_name = f'{self.coin}_{self.tf}_{self.exchange}_{self.instrument}.csv'
        logger.info(f"READ: {price_file_name}")
        metric_df = pd.read_csv(
            os.path.join(self.PRICE_FOLDER_PATH, price_file_name)
        )[['timestamp', 'close']]
        metric_df = metric_df.rename(columns={'close': 'metric'})
        self.metric_df = metric_df
        return metric_df

    def get_metric_df(self) ->pd.DataFrame:
        if self.tf != '1d': tf = '1h'
        metric_file_name = f'{self.metric_name}_{self.coin}_{self.tf}.csv'
        logger.info(f"READ: {metric_file_name}")
        metric_df = pd.read_csv(
            os.path.join(self.METRIC_FOLDER_PATH, metric_file_name), index_col=0
        ).reset_index()
        metric_df = metric_df[['timestamp', self.metric_name]]
        metric_df = metric_df.rename(columns={self.metric_name: 'metric'})
        self.metric_df = metric_df
        return metric_df

    def data_preprocessing(self) -> pd.DataFrame:
        price_df = self.price_df
        metric_df = self.metric_df
        metrics_name = self.metric_name
        shift = self.shift
        START_DATE = self.START_DATE
        END_DATE = self.END_DATE

        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], utc=True)
        price_df = price_df[['timestamp', 'close']]
        price_df = price_df[(price_df['timestamp'] >= START_DATE) & (price_df['timestamp'] <= END_DATE)]

        metric_df['timestamp'] = pd.to_datetime(metric_df['timestamp'], utc=True)
        logger.debug(f'shift: {shift} bar')
        metric_df['metric'] = metric_df['metric'].shift(self.shift)  # TODO: shift 2 bar / hours
        metric_df = metric_df[(metric_df['timestamp'] > START_DATE) & (metric_df['timestamp'] <= END_DATE)]

        df = pd.DataFrame()
        df = pd.merge(price_df, metric_df, left_on='timestamp', right_on='timestamp', how='left')
        # Find the index of the first non-NaN value in the 'metric' column
        first_valid_idx = df['metric'].first_valid_index()
        # Find the index of the last non-NaN value in the 'metric' column
        last_valid_idx = df['metric'].last_valid_index()
        # Drop leading and trailing NaN values from the DataFrame
        df = df.loc[first_valid_idx:last_valid_idx]
        df = df[['timestamp', 'close', 'metric']]
        df = df.set_index('timestamp')
        if df['close'].isna().sum() > 0:
            logger.warning(f"df close na count: {df['close'].isna().sum()}")
        if df['metric'].isna().sum() > 0:
            logger.warning(f"df metric na count: {df['metric'].isna().sum()}")

        return df

    def calculate_positions(self,df: pd.DataFrame, threshold: float) -> pd.DataFrame:

        unit_tc=self.unit_tc
        is_momentum = self.is_momentum

        df['position'] = 0
        df['ret'] = 0
        df['strategy_ret'] = 0
        df['cum_ret'] = 0
        df['transaction_cost'] = 0
        df['strategy_drawdown'] = 0
        df['benchmark_drawdown'] = 0

        df = self.assign_position(df, threshold)
        df['ret'] = df['close'].pct_change()
        df['benchmark_ret'] = df['close'].pct_change()
        df['cum_benchmark_ret'] = df['benchmark_ret'].cumsum()
        logger.debug(f"unit_tc: {unit_tc}")
        df['transaction_cost'] = abs(df['position'] - df['position'].shift(1)) * unit_tc
        df['strategy_ret'] = df['position'].shift(1) * df['ret'] - df['transaction_cost']
        df['cum_strategy_ret'] = df['strategy_ret'].cumsum()
        df['cum_strategy_ret'] = df['cum_strategy_ret'].fillna(0)

        # Calculate the strategy drawdown series
        df['strategy_cumulative_max'] = df['cum_strategy_ret'].cummax()
        df['strategy_drawdown'] = df['cum_strategy_ret'] - df['strategy_cumulative_max']

        # Calculate the benchmark drawdown series
        df['benchmark_cumulative_max'] = df['cum_benchmark_ret'].cummax()
        df['benchmark_drawdown'] = df['cum_benchmark_ret'] - df['benchmark_cumulative_max']

        return df

    def compute_kpi(self,df: pd.DataFrame) -> dict:
        kpi_dict = {}
        annualized_factor = 365 * 24 / self.tf_int
        logger.info(f"annualized factor: {annualized_factor}")

        strategy_sharpe = compute_sharpe_ratio(df['strategy_ret'], annualized_factor=annualized_factor)
        benchmark_sharpe = compute_sharpe_ratio(df['benchmark_ret'], annualized_factor=annualized_factor)

        strategy_mdd = compute_maximum_drawdown(df['strategy_ret'])
        benchmark_mdd = compute_maximum_drawdown(df['benchmark_ret'])

        strategy_ar = compute_annualized_return(df['strategy_ret'], annualized_factor=annualized_factor)
        benchamrk_ar = compute_annualized_return(df['benchmark_ret'], annualized_factor=annualized_factor)

        strategy_calmar = compute_calmar_ratio(df['strategy_ret'], annualized_factor=annualized_factor)
        benchamrk_calmar = compute_calmar_ratio(df['benchmark_ret'], annualized_factor=annualized_factor)

        strategy_recovery = compute_max_recovery_time(df['strategy_ret'])
        benchmark_recovery = compute_max_recovery_time(df['benchmark_ret'])

        strategy_t = perform_t_test(df['strategy_ret'], df['benchmark_ret'])

        alpha, beta = calculate_alpha_beta(df['strategy_ret'], df['benchmark_ret'])

        ############################################
        if (strategy_sharpe is not None):
            if (strategy_sharpe > benchmark_sharpe):
                logger.success(f"Sharpe: strategy={strategy_sharpe} || benchmark={benchmark_sharpe}")
                logger.success(f"MDD: strategy={strategy_mdd} || benchmark={benchmark_mdd}")

        ############################################
        expected_return, win_rate, win_pnl, lose_pnl = self.compute_expected_return_(df)
        long_dom = compute_long_dominance_by_duration(df, style='long&short')
        trading_number: float = abs(df['position']).sum()
        trading_freq: float = trading_number / len(df)

        position_counts = df['position'].value_counts().to_dict()
        # Create a mapping for the key replacements
        key_mapping = {
            0.0: 'no position',
            1.0: 'long',
            -1.0: 'short'
        }
        # Replace the keys in the position_counts dictionary
        position_counts = {key_mapping.get(key, key): value for key, value in position_counts.items()}
        # Calculate the sum of the values
        total_count = sum(position_counts.values())
        # Normalize the values by dividing by the sum and round to 4 decimal places
        position_counts_normalized = {
            key: Decimal(str(value / total_count)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
            for key, value in position_counts.items()
        }

        kpi_dict = {
            'strategy_sharpe': strategy_sharpe,
            'benchmark_sharpe': benchmark_sharpe,

            'strategy_mdd': strategy_mdd,
            'benchmark_mdd': benchmark_mdd,

            'strategy_ar': strategy_ar,
            'benchamrk_ar': benchamrk_ar,

            'strategy_calmar': strategy_calmar,
            'benchamrk_calmar': benchamrk_calmar,

            'strategy_recovery': strategy_recovery,
            'benchmark_recovery': benchmark_recovery,

            'strategy_t': strategy_t,
            'alpha': alpha,
            'beta': beta,
            ############################################
            'EV': expected_return,
            'win_rate': win_rate,
            'win_pnl': win_pnl,
            'lose_pnl': lose_pnl,

            'time_in_market': 1 - (position_counts_normalized.get('no position') or 0),
            'long_dominance_by_duration': long_dom,
        }
        if position_counts_normalized.get('no position') != 1:
            try:
                kpi_dict['long_dominance_count'] = round(position_counts_normalized.get('long') / (
                            position_counts_normalized.get('long') + position_counts_normalized.get('short')), 4)
            except:
                kpi_dict['long_dominance_count'] = 0
        else:
            kpi_dict['long_dominance_count'] = 0
        kpi_dict['trading_number'] = trading_number
        kpi_dict['trading_freq'] = trading_freq
        kpi_dict['duration'] = len(df)

        return kpi_dict

    def backtesting(self, isPlot: bool = False, train_size: float = 0.7,
                    is_save_ts: bool = False):

        coin = self.coin
        tf = self.tf
        metric_type = self.metric_type
        metrics_name = self.metric_name
        lookback_period = self.lookback_period
        threshold = self.threshold
        isMomentum = self.is_momentum
        denoise_logic = self.denoise_logic
        shift = self.shift
        unit_tc = self.unit_tc

        tf_int = self.tf_int
        tf_resample = self.tf.upper()
        exchange = self.exchange
        instrument = self.instrument
        price_folder_path = self.PRICE_FOLDER_PATH
        metric_folder_path = self.METRIC_FOLDER_PATH

        if shift is None:
            if metric_type == 'price':
                shift = 1
            elif metric_type == 'onchain':
                shift = 2
            else:
                raise ValueError(f"metric_type: {metric_type} is not supported")

        price_df = self.get_price_df()

        if metric_type == 'price':
            metric_df = self.get_price_metric_df()
        elif metric_type == 'onchain':
            metric_df = self.get_metric_df()
        else:
            raise ValueError(f"metric_type: {metric_type} is not supported")

        df_clean = self.data_preprocessing()

        split_timestamp = df_clean.index[int(len(df_clean) * train_size)]
        train_df = df_clean[:split_timestamp]
        test_df = df_clean[split_timestamp:]

        df_denoise = self.denoise(df=df_clean, lookback_period=lookback_period).dropna()
        df_denoise_train = self.denoise(df=train_df, lookback_period=lookback_period).dropna()
        df_denoise_test = self.denoise(df=test_df,  lookback_period=lookback_period).dropna()

        df_denoise = self.calculate_positions(df_denoise, threshold)
        df_denoise_train = self.calculate_positions(df_denoise_train, threshold)
        df_denoise_test = self.calculate_positions(df_denoise_test, threshold)

        if is_save_ts:
            file_name = f'{metrics_name}|{coin}|{tf}|{lookback_period}|{threshold}|{isMomentum}|{denoise_logic}.csv'
            time_series_folder_path = os.path.join('../main/result', metrics_name, denoise_logic, 'time_series')

            if not os.path.exists(time_series_folder_path):
                os.makedirs(time_series_folder_path)
            df_denoise.to_csv(os.path.join(time_series_folder_path, file_name.replace('.csv', '_full.csv')))
            df_denoise_train.to_csv(os.path.join(time_series_folder_path, file_name.replace('.csv', '_train.csv')))
            df_denoise_test.to_csv(os.path.join(time_series_folder_path, file_name.replace('.csv', '_test.csv')))

        train_kpi_dict = self.compute_kpi(df_denoise_train)
        test_kpi_dict = self.compute_kpi(df_denoise_test)
        full_set_dict = self.compute_kpi(df_denoise)

        train_kpi_dict['set'] = 'train'
        test_kpi_dict['set'] = 'test'
        full_set_dict['set'] = 'full'

        for i in [train_kpi_dict, test_kpi_dict, full_set_dict]:
            i['shift'] = shift
            i['unit_tc'] = unit_tc
            i['annulaized_factor'] = 365 * 24 / tf_int

        if isPlot:
            # # Plot for df_denoise_train
            # plt.figure(figsize=(10, 6))
            # df_denoise_train[['cum_strategy_ret', 'cum_benchmark_ret']].plot()
            # plt.title('Cumulative Returns - Train Set')
            # plt.xlabel('Date')
            # plt.ylabel('Cumulative Returns')
            # plt.legend(['Strategy Returns', 'Benchmark Returns'])
            # plt.tight_layout()
            # # Plot for df_denoise_test
            # plt.figure(figsize=(10, 6))
            # df_denoise_test[['cum_strategy_ret', 'cum_benchmark_ret']].plot()
            # plt.title('Cumulative Returns - Test Set')
            # plt.xlabel('Date')
            # plt.ylabel('Cumulative Returns')
            # plt.legend(['Strategy Returns', 'Benchmark Returns'])
            # plt.tight_layout()
            # # Plot for df_denoise
            # plt.figure(figsize=(10, 6))
            # df_denoise[['cum_strategy_ret', 'cum_benchmark_ret']].plot()
            # plt.title('Cumulative Returns - Full Dataset')
            # plt.xlabel('Date')
            # plt.ylabel('Cumulative Returns')
            # plt.legend(['Strategy Returns', 'Benchmark Returns'])
            # plt.tight_layout()

            # plt.show()

            fig = make_subplots(rows=3, cols=3, subplot_titles=(
                'Return Time Series - Train', 'Return Time Series - Test', 'Return Time Series - Full',
                'Cumulative Returns - Train', 'Cumulative Returns - Test', 'Cumulative Returns - Full',
                'Drawdown - Train', 'Drawdown - Test', 'Drawdown - Full'))

            # Return Time Series - Train
            fig.add_trace(
                go.Scatter(x=df_denoise_train.index, y=df_denoise_train['strategy_ret'], name='Strategy Returns',
                           line=dict(color='blue')),
                row=1, col=1)
            fig.add_trace(
                go.Scatter(x=df_denoise_train.index, y=df_denoise_train['benchmark_ret'], name='Benchmark Returns',
                           line=dict(color='orange')),
                row=1, col=1)

            # Return Time Series - Test
            fig.add_trace(
                go.Scatter(x=df_denoise_test.index, y=df_denoise_test['strategy_ret'], name='Strategy Returns',
                           line=dict(color='blue')),
                row=1, col=2)
            fig.add_trace(
                go.Scatter(x=df_denoise_test.index, y=df_denoise_test['benchmark_ret'], name='Benchmark Returns',
                           line=dict(color='orange')),
                row=1, col=2)

            # Return Time Series - Full
            fig.add_trace(go.Scatter(x=df_denoise.index, y=df_denoise['strategy_ret'], name='Strategy Returns',
                                     line=dict(color='blue')),
                          row=1, col=3)
            fig.add_trace(go.Scatter(x=df_denoise.index, y=df_denoise['benchmark_ret'], name='Benchmark Returns',
                                     line=dict(color='orange')),
                          row=1, col=3)

            # Cumulative Returns - Train
            fig.add_trace(
                go.Scatter(x=df_denoise_train.index, y=df_denoise_train['cum_strategy_ret'], name='Strategy Returns',
                           line=dict(color='blue')),
                row=2, col=1)
            fig.add_trace(
                go.Scatter(x=df_denoise_train.index, y=df_denoise_train['cum_benchmark_ret'], name='Benchmark Returns',
                           line=dict(color='orange')),
                row=2, col=1)

            # Cumulative Returns - Test
            fig.add_trace(
                go.Scatter(x=df_denoise_test.index, y=df_denoise_test['cum_strategy_ret'], name='Strategy Returns',
                           line=dict(color='blue')),
                row=2, col=2)
            fig.add_trace(
                go.Scatter(x=df_denoise_test.index, y=df_denoise_test['cum_benchmark_ret'], name='Benchmark Returns',
                           line=dict(color='orange')),
                row=2, col=2)

            # Cumulative Returns - Full
            fig.add_trace(go.Scatter(x=df_denoise.index, y=df_denoise['cum_strategy_ret'], name='Strategy Returns',
                                     line=dict(color='blue')),
                          row=2, col=3)
            fig.add_trace(go.Scatter(x=df_denoise.index, y=df_denoise['cum_benchmark_ret'], name='Benchmark Returns',
                                     line=dict(color='orange')),
                          row=2, col=3)

            # Drawdown - Train
            fig.add_trace(
                go.Scatter(x=df_denoise_train.index, y=df_denoise_train['strategy_drawdown'], name='Strategy Drawdown',
                           line=dict(color='blue')),
                row=3, col=1)
            fig.add_trace(
                go.Scatter(x=df_denoise_train.index, y=df_denoise_train['benchmark_drawdown'],
                           name='Benchmark Drawdown',
                           line=dict(color='orange')),
                row=3, col=1)

            # Drawdown - Test
            fig.add_trace(
                go.Scatter(x=df_denoise_test.index, y=df_denoise_test['strategy_drawdown'], name='Strategy Drawdown',
                           line=dict(color='blue')),
                row=3, col=2)
            fig.add_trace(
                go.Scatter(x=df_denoise_test.index, y=df_denoise_test['benchmark_drawdown'], name='Benchmark Drawdown',
                           line=dict(color='orange')),
                row=3, col=2)

            # Drawdown - Full
            fig.add_trace(go.Scatter(x=df_denoise.index, y=df_denoise['strategy_drawdown'], name='Strategy Drawdown',
                                     line=dict(color='blue')),
                          row=3, col=3)
            fig.add_trace(go.Scatter(x=df_denoise.index, y=df_denoise['benchmark_drawdown'], name='Benchmark Drawdown',
                                     line=dict(color='orange')),
                          row=3, col=3)

            # Update layout
            fig.update_layout(height=1080, width=1920,
                              title_text=f"Strategy vs Benchmark Performance (Training size: {train_size}) || params: {coin} | {tf} | {metrics_name} | {lookback_period} | {threshold} | {denoise_logic} | {isMomentum}",
                              legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1))
            fig.show()

        pio.write_html(fig,
                       f"{self.RESULT_FOLDER_PATH}/{denoise_logic}/time_series_{denoise_logic}_{isMomentum}_{lookback_period}_{threshold}.html")
        return train_kpi_dict, test_kpi_dict, full_set_dict

    def optimization(self,asset_list: list,
                     tf_list: list, lookback_period_list: list,
                     isMomentum_list: list, denoise_logic_list: list, threshold_list: Optional[List] = None,
                     is_save_ts: bool = False,is_plot: bool = False,train_size=0.7):

        metrics_name = self.metric_name
        metric_type = self.metric_type
        shift = self.shift
        unit_tc = self.unit_tc



        if threshold_list is None:
            if denoise_logic_list == ['z'] :
                threshold_list = np.arange(0.1, 3.0, 0.1)
            elif denoise_logic_list == ['pct'] or denoise_logic_list == ['ma_diff']:
                threshold_list = np.arange(0.00, 0.2, 0.0025)

            logger.debug('Optimization with default threshold_list')

        # Create a list of all hyperparameter combinations
        hyperparameter_combinations = list(
            itertools.product(asset_list, tf_list, isMomentum_list, lookback_period_list, threshold_list,
                              denoise_logic_list))
        # print(pd.DataFrame(hyperparameter_combinations))
        logger.info(f"Total number of hyperparameter combinations: {len(hyperparameter_combinations)}")

        # exchange = 'binance'
        # instrument = 'perp'
        # price_folder_path = '../Data/Price'
        # metric_folder_path = f'../Data/OnChain'

        exchange = self.exchange
        instrument = self.instrument
        price_folder_path = self.PRICE_FOLDER_PATH
        metric_folder_path = self.METRIC_FOLDER_PATH

        kpi_list = []
        train_kpi_dict_list = []
        test_kpi_dict_list = []
        full_set_dict_list = []

        # Iterate over each hyperparameter combination and perform backtesting
        for coin, tf, isMomentum, lookback_period, threshold, denoise_logic in hyperparameter_combinations:
            print('=' * 200)
            threshold = Decimal(str(threshold)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
            logger.info(
                f'coin:{coin}| tf:{tf}| isMomentum:{isMomentum}| lookback_period:{lookback_period}| threshold:{threshold}| denoise_logic:{denoise_logic}')

            if os.path.join(metrics_name, denoise_logic) not in os.listdir('../main/result'):
                result_folder_path = os.path.join('../main/result', metrics_name, denoise_logic)
                if not os.path.exists(result_folder_path):
                    os.makedirs(result_folder_path)
                    logger.debug(f"Create folder: {result_folder_path}")
                else:
                    pass
                    # logger.debug(f"Folder already exists: {result_folder_path}")

            kpi = {}
            train_kpi_dict = {}
            test_kpi_dict = {}
            full_set_dict = {}

            train_kpi_dict, test_kpi_dict, full_set_dict = self.backtesting(
                 isPlot=is_plot, train_size=train_size, is_save_ts=is_save_ts)

            train_kpi_dict['coin'] = coin
            train_kpi_dict['tf'] = tf
            train_kpi_dict['isMomentum'] = isMomentum
            train_kpi_dict['lookback_period'] = lookback_period
            train_kpi_dict['threshold'] = threshold
            train_kpi_dict['denoise_logic'] = denoise_logic
            train_kpi_dict_list.append(train_kpi_dict)

            test_kpi_dict['coin'] = coin
            test_kpi_dict['tf'] = tf
            test_kpi_dict['isMomentum'] = isMomentum
            test_kpi_dict['lookback_period'] = lookback_period
            test_kpi_dict['threshold'] = threshold
            test_kpi_dict['denoise_logic'] = denoise_logic
            test_kpi_dict_list.append(test_kpi_dict)

            full_set_dict['coin'] = coin
            full_set_dict['tf'] = tf
            full_set_dict['isMomentum'] = isMomentum
            full_set_dict['lookback_period'] = lookback_period
            full_set_dict['threshold'] = threshold
            full_set_dict['denoise_logic'] = denoise_logic
            full_set_dict_list.append(full_set_dict)

        print('+++' * 50)

        train_kpi_df = pd.DataFrame(train_kpi_dict_list).sort_values(by='strategy_sharpe', ascending=False)
        test_kpi_df = pd.DataFrame(test_kpi_dict_list).sort_values(by='strategy_sharpe', ascending=False)
        full_kpi_df = pd.DataFrame(full_set_dict_list).sort_values(by='strategy_sharpe', ascending=False)

        # return pd.DataFrame(kpi_list).sort_values(by='strategy_sharpe', ascending=False)
        return train_kpi_df, test_kpi_df, full_kpi_df

    def create_heatmap_interactive(self,
            metrics_name: str = 'BTC',
            train_test_full: str = 'train', coin: str = 'BTC',
            tf: str = '8h', isMomentum: bool = False, denoise_logic: str = 'z',
            title=None, kpi: str = 'strategy_sharpe', ):

        result_folder_path = f'result/{metrics_name}'
        opt_df = pd.read_csv(f'{result_folder_path}/{denoise_logic}/performance_{denoise_logic}_{train_test_full}.csv',
                             index_col=0)

        df = opt_df[
            (opt_df.coin == coin) &
            (opt_df.tf == tf) &
            (opt_df.isMomentum == isMomentum) &
            (opt_df.denoise_logic == denoise_logic)
            ]
        heatmap_data = df.pivot(index='lookback_period', columns='threshold', values=kpi)
        annotations = heatmap_data.applymap(lambda x: f'{x:.2f}' if not pd.isnull(x) else '')
        custom_colorscale = [[0, 'red'],
                             [0.5, 'white'],
                             [0.8, 'orange'],
                             [1, 'green']]
        fig = ff.create_annotated_heatmap(
            z=heatmap_data.to_numpy(),
            x=heatmap_data.columns.tolist(),
            y=heatmap_data.index.tolist(),
            annotation_text=annotations.to_numpy(),
            colorscale=custom_colorscale,
            zmid=0.0,
            hoverinfo='z',
            showscale=True,
        )
        if title:
            header = title
        else:
            header = f"{metrics_name}_{coin}_{tf}_{isMomentum}_{denoise_logic}_{kpi}_{train_test_full}"

        # Update layout
        fig.update_layout(
            title={
                # 'text': metrics_name+f'_{coin}_{tf}'+'_heatmap',
                'text': header,
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis=dict(
                title='threshold',
                tickmode='array',
                tickvals=heatmap_data.columns.tolist(),
                ticktext=[str(val) for val in heatmap_data.columns.tolist()],
                side='bottom'
            ),
            yaxis=dict(
                title='lookback_period',
                tickmode='array',
                tickvals=heatmap_data.index.tolist(),
                ticktext=[str(val) for val in heatmap_data.index.tolist()]
            ),
            # width=1920,
            # height=1080,
            autosize=True
        )
        # Here we add a custom hovertemplate
        for i in range(len(fig.data)):
            fig.data[i].hovertemplate = 'X: %{x}<br>Y: %{y}<br>value: %{z}<extra></extra>'

        fig.update_xaxes(side='bottom')
        # Save the interactive plot as an HTML file
        pio.write_html(fig,
                       f"{result_folder_path}/{denoise_logic}/heatmap_{denoise_logic}_{kpi}_{train_test_full}.html")

        fig.write_image(f"{result_folder_path}/{denoise_logic}/heatmap_{denoise_logic}_{kpi}_{train_test_full}.png"
                        , width=2560, height=1440 * 2, scale=1)
        fig.show()

    def create_heatmap_seaborn(self, metrics_name: str = 'BTC', train_test_full: str = 'train', coin: str = 'BTC',
                       tf: str = '8h', isMomentum: bool = False, denoise_logic: str = 'z',
                       title=None, kpi: str = 'strategy_sharpe'):

        result_folder_path = f'result/{metrics_name}'
        opt_df = pd.read_csv(f'{result_folder_path}/{denoise_logic}/performance_{denoise_logic}_{train_test_full}.csv',
                             index_col=0)

        df = opt_df[
            (opt_df.coin == coin) &
            (opt_df.tf == tf) &
            (opt_df.isMomentum == isMomentum) &
            (opt_df.denoise_logic == denoise_logic)
        ]
        heatmap_data = df.pivot(index='lookback_period', columns='viridis', values=kpi)

        # Create the heatmap using seaborn
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='magma', center=0,
                         linewidths=.5, cbar_kws={"shrink": .8})

        # Set title
        if title:
            plt.title(title)
        else:
            title = f"{metrics_name}_{coin}_{tf}_{isMomentum}_{denoise_logic}_{kpi}_{train_test_full}"
            plt.title(title)

        # Set axis labels
        plt.xlabel('Threshold')
        plt.ylabel('Lookback Period')

        # Save the figure
        plt.savefig(f"{result_folder_path}/{denoise_logic}/heatmap_{denoise_logic}_{kpi}_{train_test_full}.png", dpi=300)
        plt.show()

if __name__ == '__main__':
    backtester = Backtester(
        metric_type='price',
        metric_name='BTC',
        coin='BTC',
        tf='8h',
        is_momentum=True,
        threshold=1.3,
        lookback_period=425,
        denoise_logic='z',
        shift=0,
        unit_tc=0.0,
        style='long&short',
        training_size=0.7
    )
    backtester.backtesting(isPlot=True, train_size=0.7, is_save_ts=False)
    # 
    # 
    # backtester.optimization(
    #     asset_list=['BTC'],
    #     tf_list=['8h'],
    #     lookback_period_list=np.arange(5,505,5),
    #     isMomentum_list=[True,False],
    #     denoise_logic_list=['z'],
    #     threshold_list=None,
    #     is_save_ts=False,
    #     is_plot=False,
    #     train_size=0.7
    # )

    # backtester.create_heatmap_interactive(
    #     metrics_name='BTC',
    #     train_test_full='train',
    #     coin='BTC',
    #     tf='8h',
    #     isMomentum=True,
    #     denoise_logic='z',
    #     title=None,
    #     kpi='strategy_sharpe'
    # )
