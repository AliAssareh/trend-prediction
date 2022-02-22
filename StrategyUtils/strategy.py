from FeatureEngineeringUtils.btc_feature_engineering_utils import import_processed_data, TrainTestValidationLoader
from ModelingUtils.trainers import NVAE
from ModelingUtils.models import load_fcn, predict
from enum import Enum
import numpy as np
import pandas as pd


class Decision(Enum):
    Neutral = 4
    Long = 3
    CloseShort = 2
    CloseLong = 1
    Short = 0


class Position(Enum):
    Long = 1
    Neutral = 0
    Short = -1


class Strategy(Enum):
    two_states_buy_only = 0
    four_states_buy_only = 1
    two_states_buy_and_sell = 2
    four_states_buy_and_sell = 3


def generate_predictions(time_frame):
    if time_frame == 240:
        nvae = NVAE('btc_4h_for_target1_1', 'target1', suffix=1, latent_dim=36, nca_dim=10)
    elif time_frame == 60:
        nvae = NVAE('btc_1h_for_target1_1', 'target1', suffix=1, latent_dim=32, nca_dim=10)
    elif time_frame == 15:
        nvae = NVAE('btc_15m_for_target1_1', 'target1', suffix=1, latent_dim=32, nca_dim=10)
    else:
        raise Exception('Other timeframes are not ready yet!')

    nvae.generate_autoencoder(silent=True)
    nvae.train_auto_encoder(just_load=True)
    nvae.generate_fully_connected_network()
    nvae.train_fully_connected_net(just_load=True)
    train_preds, test_preds, val_preds = nvae.generate_predictions()

    train_df = pd.DataFrame(train_preds.reshape(-1, 1), index=nvae.data_loader.x_train.index, columns=['decision'])
    test_df = pd.DataFrame(test_preds.reshape(-1, 1), index=nvae.data_loader.x_test.index, columns=['decision'])
    val_df = pd.DataFrame(val_preds.reshape(-1, 1), index=nvae.data_loader.x_val.index, columns=['decision'])
    return train_df, test_df, val_df


def get_predictions_dfs(timeframe):
    train_df, test_df, val_df = generate_predictions(timeframe)
    raw_df = import_processed_data(f'PreprocessedData//BTCUSDT_{timeframe}_1.csv',
                                   index_column_name='datetime')

    train_df['high'] = raw_df.loc[train_df.index, 'BTCUSDT_high']
    train_df['low'] = raw_df.loc[train_df.index, 'BTCUSDT_low']
    train_df['closing_price'] = raw_df.loc[train_df.index, 'BTCUSDT_close']

    test_df['high'] = raw_df.loc[test_df.index, 'BTCUSDT_high']
    test_df['low'] = raw_df.loc[test_df.index, 'BTCUSDT_low']
    test_df['closing_price'] = raw_df.loc[test_df.index, 'BTCUSDT_close']

    val_df['high'] = raw_df.loc[val_df.index, 'BTCUSDT_high']
    val_df['low'] = raw_df.loc[val_df.index, 'BTCUSDT_low']
    val_df['closing_price'] = raw_df.loc[val_df.index, 'BTCUSDT_close']

    return train_df, test_df, val_df


def save_strategy_df(decision_df, name):
    decision_df.to_csv(f'StrategyData//{name}.csv')


def load_strategy_df(name, index_column_name='datetime'):
    df = import_processed_data(f'StrategyData//{name}.csv', index_column_name=index_column_name)
    return df


class StrategyTester:
    def __init__(self, decisions_df):
        self.check_the_columns_of_df(decisions_df)
        self.length_of_df = self.check_the_length_of_df(decisions_df)
        self.df = decisions_df
        self.pnl_df = pd.DataFrame(index=self.df.index, columns=['pnl', 'live_cpnl', 'btc_live_cpnl'])
        self.cpnl = [0]
        self.btc_cpnl = [self.df.iloc[0, self.df.columns.get_loc('closing_price')]]
        self.highest_cpnl = 0
        self.mdd = 0
        self.position = Position.Neutral.value
        self.position_price = None
        self.n_positions = 0

    def generate_results(self, strategy=Strategy.four_states_buy_and_sell.value):
        self._reset_the_variables()
        generator = self.generate_date_price_and_decision()
        if strategy == Strategy.two_states_buy_only.value:
            self._two_states_buy_only(generator)
        elif strategy == Strategy.four_states_buy_only.value:
            self._four_states_buy_only(generator)
        elif strategy == Strategy.two_states_buy_and_sell.value:
            self._two_states_buy_and_sell(generator)
        elif strategy == Strategy.four_states_buy_and_sell.value:
            self._four_states_buy_and_sell(generator)
        else:
            raise Exception('this strategy is not implemented yet!')
        self.pnl_df['close'] = self.df.closing_price
        return self.pnl_df.copy()

    @staticmethod
    def check_the_length_of_df(df):
        length_of_df = len(df)
        if length_of_df < 10:
            raise Exception('Decision dataframe should be longer for evaluation of results.')
        else:
            return length_of_df

    @staticmethod
    def check_the_columns_of_df(df):
        if 'decision' not in df.columns.to_list():
            raise Exception('decision_df should contain decision column.')
        elif 'high' not in df.columns.to_list():
            raise Exception('decision_df should contain high column.')
        elif 'low' not in df.columns.to_list():
            raise Exception('decision_df should contain low column.')
        elif 'closing_price' not in df.columns.to_list():
            raise Exception('decision_df should contain closing_price column.')
        else:
            return True

    def _reset_the_variables(self):
        self.pnl_df = pd.DataFrame(index=self.df.index, columns=['pnl', 'live_cpnl', 'btc_live_cpnl'])
        self.cpnl = [0]
        self.btc_cpnl = [self.df.iloc[0, self.df.columns.get_loc('closing_price')]]
        self.highest_cpnl = 0
        self.mdd = 0
        self.position = Position.Neutral.value
        self.position_price = None
        self.n_positions = 0

    def _two_states_buy_only(self, generator):
        for date, prices, decision in generator:
            self._evaluate_the_mdd_and_live_cpnl(date, prices)
            if decision == Decision.Long.value:
                self._long_signal(prices[-1], date)
            elif decision == Decision.Short.value and self.position == Position.Long.value:
                self._close_long_signal(prices[-1], date)
            else:
                pass
        self._if_there_is_any_open_position_close_it()

    def _four_states_buy_only(self, generator):
        for date, prices, decision in generator:
            self._evaluate_the_mdd_and_live_cpnl(date, prices)
            if decision == Decision.Long.value:
                self._long_signal(prices[-1], date)
            elif decision == Decision.CloseLong.value and self.position == Position.Long.value:
                self._close_long_signal(prices[-1], date)
            elif decision == Decision.Short.value and self.position == Position.Long.value:
                self._close_long_signal(prices[-1], date)
            else:
                pass
        self._if_there_is_any_open_position_close_it()

    def _two_states_buy_and_sell(self, generator):
        for date, prices, decision in generator:
            self._evaluate_the_mdd_and_live_cpnl(date, prices)
            if decision == Decision.Long.value:
                self._long_signal(prices[-1], date)
            elif decision == Decision.Short.value:
                self._short_signal(prices[-1], date)
            else:
                pass
        self._if_there_is_any_open_position_close_it()

    def _four_states_buy_and_sell(self, generator):
        for date, prices, decision in generator:
            self._evaluate_the_mdd_and_live_cpnl(date, prices)
            if decision == Decision.Long.value:
                self._long_signal(prices[-1], date)
            elif decision == Decision.CloseLong.value and self.position == Position.Long.value:
                self._close_long_signal(prices[-1], date)
            elif decision == Decision.CloseShort.value and self.position == Position.Short.value:
                self._close_short_signal(prices[-1], date)
            elif decision == Decision.Short.value:
                self._short_signal(prices[-1], date)
            else:
                pass
        self._if_there_is_any_open_position_close_it()

    def generate_date_price_and_decision(self):
        for i in range(self.length_of_df - 1):
            row = self.df.iloc[i]
            date = row.name.to_pydatetime()
            yield date, [row.high, row.low, row.closing_price], row.decision

    def _long_signal(self, price, date):
        if self.position == Position.Neutral.value:
            self._open_long_position(price)
        elif self.position == Position.Short.value:
            self._close_current_position(price, date)
            self._open_long_position(price)

    def _close_long_signal(self, price, date):
        if self.position == Position.Long.value:
            self._close_current_position(price, date)
        else:
            raise Exception('There is no open long position to close.')

    def _short_signal(self, price, date):
        if self.position == Position.Long.value:
            self._close_current_position(price, date)
            self._open_short_position(price)
        elif self.position == Position.Neutral.value:
            self._open_short_position(price)

    def _close_short_signal(self, price, date):
        if self.position == Position.Short.value:
            self._close_current_position(price, date)
        else:
            raise Exception('There is no open short position to close.')

    def _open_long_position(self, price):
        self.position = Position.Long.value
        self.position_price = price

    def _open_short_position(self, price):
        self.position = Position.Short.value
        self.position_price = price

    def _if_there_is_any_open_position_close_it(self):
        row = self.df.iloc[-1]
        price = row.closing_price
        date = row.name.to_pydatetime()
        if self.position != Position.Neutral.value:
            self._close_current_position(price, date)

    def _close_current_position(self, price, date):
        if self.position == Position.Long.value:
            pnl = 100 * (price - self.position_price) / self.position_price
            btc_pnl = price - self.position_price
            self.pnl_df.loc[date, 'pnl'] = pnl
            self.cpnl.append(self.cpnl[-1] + pnl)
            self.btc_cpnl.append(self.btc_cpnl[-1] + btc_pnl)
            self.position = Position.Neutral.value
            self.n_positions = self.n_positions + 1
        elif self.position == Position.Short.value:
            negative_pnl = 100 * (price - self.position_price) / self.position_price
            negative_btc_pnl = price - self.position_price
            self.pnl_df.loc[date, 'pnl'] = -negative_pnl
            self.cpnl.append(self.cpnl[-1] - negative_pnl)
            self.btc_cpnl.append(self.btc_cpnl[-1] - negative_btc_pnl)
            self.position = Position.Neutral.value
            self.n_positions = self.n_positions + 1
        else:
            pass

    def _evaluate_the_mdd_and_live_cpnl(self, date, prices):
        if self.position == Position.Neutral.value:
            pass
        else:
            if self.position == Position.Long.value:
                max_probable_loss = np.minimum(100 * (prices[1] - self.position_price) / self.position_price, 0)
                live_cpnl = self.cpnl[-1] + 100 * (prices[-1] - self.position_price) / self.position_price
                self.pnl_df.loc[date, 'live_cpnl'] = live_cpnl
                self.pnl_df.loc[date, 'btc_live_cpnl'] = self.btc_cpnl[-1] + (prices[-1] - self.position_price)
            else:
                max_probable_loss = np.minimum(-100 * ((prices[0] - self.position_price) / self.position_price), 0)
                live_cpnl = self.cpnl[-1] - 100 * (prices[-1] - self.position_price) / self.position_price
                self.pnl_df.loc[date, 'live_cpnl'] = live_cpnl
                self.pnl_df.loc[date, 'btc_live_cpnl'] = self.btc_cpnl[-1] - (prices[-1] - self.position_price)
            self.highest_cpnl = np.maximum(self.highest_cpnl, live_cpnl)
            draw_down = np.maximum(self.highest_cpnl - (self.cpnl[-1] + max_probable_loss), 0)
            self.mdd = np.maximum(self.mdd, draw_down)
