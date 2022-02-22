from sklearn.metrics import r2_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from Indicators.stationary_indicators import add_all_stationary_ta_features
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from ta import add_all_ta_features
from pykalman import KalmanFilter
from enum import Enum
import numpy as np
import pandas as pd
import os


class Target(Enum):
    target1 = 1
    target2 = 2
    target3 = 3


class Exogenous(Enum):
    none = 0
    regular = 1
    midas = 2
    only = 3


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


def import_raw_data(address, index_column_name='datetime', time_format="%Y-%m-%d %H:%M:%S", from_unix=False):
    df = pd.read_csv(address)
    if from_unix:
        df['datetime'] = pd.to_datetime(df[index_column_name], unit='s')
    else:
        df['datetime'] = pd.to_datetime(df[index_column_name], format=time_format)
    df.sort_values('datetime', inplace=True)
    df.set_index('datetime', inplace=True)
    if index_column_name != 'datetime':
        df.drop(columns=[index_column_name], inplace=True)
    if 'up_first' in df.columns:
        if 'volume' in df.columns:
            df = df.loc[:, ['open', 'high', 'low', 'close', 'volume', 'up_first']]
        else:
            df = df.loc[:, ['open', 'high', 'low', 'close', 'up_first']]
    else:
        if 'volume' in df.columns:
            df = df.loc[:, ['open', 'high', 'low', 'close', 'volume']]
        else:
            df = df.loc[:, ['open', 'high', 'low', 'close']]
    return df


def import_processed_data(address, index_column_name='datetime', time_format="%Y-%m-%d %H:%M:%S"):
    df = pd.read_csv(address)
    df['datetime'] = pd.to_datetime(df[index_column_name], format=time_format)
    if index_column_name != 'datetime':
        df.drop(columns=[index_column_name], inplace=True)
    df.sort_values('datetime', inplace=True)
    df.set_index('datetime', inplace=True)
    return df


def preprocess_m1_df(address, index_column_name='datetime'):
    raw_m1_df = import_raw_data(address, index_column_name)
    if os.path.isfile('PreprocessedData//BTCUSDT_1.csv'):
        preprocessed_m1_df = import_processed_data('PreprocessedData//BTCUSDT_1.csv', index_column_name='datetime')
        last_stamp = preprocessed_m1_df.tail(1).index.to_pydatetime()[0]
        last_stamp = last_stamp - timedelta(hours=last_stamp.hour, minutes=last_stamp.minute, seconds=last_stamp.second)
        rest_of_raw_m1 = raw_m1_df.loc[last_stamp:]
        if len(rest_of_raw_m1) > 1:
            m1_linear_interpolator = LinearInterpolator(rest_of_raw_m1, time_frame_in_minutes=1,
                                                        first_day=datetime.strftime(last_stamp, '%Y-%m-%d %H:%M:%S'))
            m1_df = m1_linear_interpolator.return_complete_dataframe()
            preprocessed_m1_df.drop(preprocessed_m1_df.loc[last_stamp:].index, inplace=True)
            m1_df = pd.concat([preprocessed_m1_df, m1_df], axis=0)
            m1_df.index.name = 'datetime'
            m1_df.sort_index(inplace=True)
            m1_df.to_csv('PreprocessedData//BTCUSDT_1.csv')
        else:
            print('BTCUSDT_1 already is uptodate')
    else:
        print('BTCUSDT_1.csv not found at PreprocessedData, preprocessing the whole series might take a while!')
        m1_linear_interpolator = LinearInterpolator(raw_m1_df, time_frame_in_minutes=1)
        m1_df = m1_linear_interpolator.return_complete_dataframe()
        m1_df.to_csv('PreprocessedData//BTCUSDT_1.csv')


class BtcPreprocessor:
    def __init__(self, original_timeframe, timeframe, use_stationary_ta=True,
                 first_day='2017-08-17 00:00:00'):
        self.df = import_raw_data(f"Data//BTCUSDT_{timeframe}.csv", index_column_name='datetime')
        self.timeframe = timeframe
        self.number_of_candles_equivalent_to_one_original_candle = int(original_timeframe / self.timeframe)
        self.use_stationary_ta = use_stationary_ta
        self.colprefix = 'BTCUSDT_'
        self.first_day = first_day
        self.features_df = None

    def run(self):
        processed_df = self.interpolate_the_raw_data_and_add_up_first()
        alt_candles_df = self._create_alternative_candles(processed_df)
        indicators_df = self._generate_indicators(processed_df)
        self.features_df = self._create_features_df(alt_candles_df, indicators_df)
        self.save_results()

    def interpolate_the_raw_data_and_add_up_first(self):
        linear_interpolator = LinearInterpolator(self.df, self.timeframe, first_day=self.first_day)
        interpolated_df = linear_interpolator.return_complete_dataframe()
        up_first_detector = UpFirstDetector(interpolated_df, self.timeframe, first_day=self.first_day)
        return up_first_detector.return_new_df()

    def _create_alternative_candles(self, processed_df):
        temp_df = processed_df.copy()
        temp_df['previous_close'] = temp_df.close.shift(+1)
        temp_df['previous_open'] = temp_df.open.shift(+1)

        first_open = temp_df.open.iloc[0].copy()
        temp_df.iloc[0, temp_df.columns.get_loc('previous_close')] = first_open
        temp_df.iloc[0, temp_df.columns.get_loc('previous_open')] = first_open

        temp_df[self.colprefix + 'ho_percent'] = (temp_df.high - temp_df.open) / temp_df.open
        temp_df[self.colprefix + 'co_percent'] = (temp_df.close - temp_df.open) / temp_df.open
        temp_df[self.colprefix + 'oc_percent'] = (temp_df.open - temp_df.previous_close) / temp_df.previous_close
        temp_df[self.colprefix + 'lo_percent'] = (temp_df.low - temp_df.open) / temp_df.open

        temp_df.rename({'open': self.colprefix + 'open', 'high': self.colprefix + 'high',
                        'low': self.colprefix + 'low', 'close': self.colprefix + 'close',
                        'volume': self.colprefix + 'volume', 'up_first': self.colprefix + 'up_first'},
                       axis=1, inplace=True)

        temp_df.drop(columns=['previous_open', 'previous_close'], inplace=True)

        return temp_df

    def _generate_indicators(self, processed_df):
        temp_df = processed_df.copy()
        if self.use_stationary_ta:
            indicators_df = add_all_stationary_ta_features(temp_df, 'open', 'high', 'low', 'close', 'volume',
                                                           'up_first', colprefix=self.colprefix)
            indicators_df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'up_first'], inplace=True)
        else:
            indicators_df = add_all_ta_features(temp_df, 'open', 'high', 'low', 'close', 'volume',
                                                colprefix=self.colprefix)
            indicators_df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'trend_psar_up',
                                        'trend_psar_down', 'up_first'], inplace=True)
        return indicators_df

    @staticmethod
    def _create_features_df(alt_candles_df, indicators_df):
        df = pd.concat([alt_candles_df, indicators_df], axis=1)
        return df

    def _drop_the_formerly_repaired_nan_values_for_given_target(self, df, target):
        n = self.number_of_candles_equivalent_to_one_original_candle
        size = 70 if n == 1 else (71 * n) - 1
        df.drop(df.head(size).index, inplace=True)
        if target in ['target1', 'target2']:
            df.drop(df.tail(n).index, inplace=True)
        else:
            df.drop(df.tail(5 * n).index, inplace=True)
        return df

    def save_results(self):
        self.features_df.to_csv(f'PreprocessedData//BTCUSDT_{self.timeframe}.csv')
        target1_df = self._drop_the_formerly_repaired_nan_values_for_given_target(self.features_df.copy(), 'target1')
        target1_df.to_csv(f'PreprocessedData//BTCUSDT_{self.timeframe}_1.csv')
        target3_df = self._drop_the_formerly_repaired_nan_values_for_given_target(self.features_df.copy(), 'target3')
        target3_df.to_csv(f'PreprocessedData//BTCUSDT_{self.timeframe}_3.csv')


class LinearInterpolator:
    def __init__(self, df: pd.DataFrame, time_frame_in_minutes=1440, first_day='2017-08-17 00:00:00',
                 last_day='2022-01-01 00:00:01', causal=False, debug=False):
        self.df = df.copy()
        self.df_columns = self.df.columns.to_list()
        self.colprefix = self.df_columns[0].split('_')[0]
        self.time_frame = time_frame_in_minutes
        self.number_of_candles_in_each_day = int(1440 / time_frame_in_minutes)
        self.first_day = datetime.strptime(first_day, '%Y-%m-%d %H:%M:%S')
        self.last_day = datetime.strptime(last_day, '%Y-%m-%d %H:%M:%S')
        self.total_number_of_missing_candles = 0
        self.number_of_candles_with_bad_index = 0
        self.number_of_candles_with_two_index = 0
        self.causal = causal
        if debug:
            self._run_with_comment()
        else:
            self._run()

    def _run_with_comment(self):
        self._assert_existence_of_last_candle()
        self._clip_the_dataframe()
        self._adjust_dataframe_indices()
        self._reset_first_candle()
        date = self.first_day
        while date < self.last_day:
            if self._given_day_has_missing_data(date):
                print(f'At the day: {date.strftime("%Y-%m-%d")}\n')
                number_of_missing_candles = 0
                if date.date() == self.last_day.date():
                    gen = self._list_of_last_days_candles(date)
                else:
                    gen = self._list_of_daily_candles(date)
                for start_of_candle in gen:
                    if start_of_candle not in self.df.index:
                        n = self._impute_the_missing_candle(start_of_candle)
                        number_of_missing_candles += n

                print(
                    f'\n{number_of_missing_candles} data {"is" if number_of_missing_candles == 1 else "are"} '
                    f'missing.The repaird data is shown below:\n')
                # print(self.df.loc[date.strftime('%Y-%m-%d')])
                print('\n\n')
            date = date + timedelta(days=1)
        self._make_sure_no_where_volume_is_zero_or_negative()
        self.df.sort_index(inplace=True)
        print('\n total number of missing data: ', self.total_number_of_missing_candles)

    def _run(self):
        self._assert_existence_of_last_candle()
        self._clip_the_dataframe()
        self._adjust_dataframe_indices()
        self._reset_first_candle()
        date = self.first_day
        while date < self.last_day:
            if self._given_day_has_missing_data(date):
                if date.date() == self.last_day.date():
                    gen = self._list_of_last_days_candles(date)
                else:
                    gen = self._list_of_daily_candles(date)
                for start_of_candle in gen:
                    if start_of_candle not in self.df.index:
                        self._impute_the_missing_candle(start_of_candle)
            date = date + timedelta(days=1)
        if 'volume' in self.df_columns:
            self._make_sure_no_where_volume_is_zero_or_negative()
        self.df.sort_index(inplace=True)

    def _assert_existence_of_last_candle(self):
        if self.last_day - timedelta(seconds=1) not in self.df.index.tolist():
            raise Exception('last candle must exist in the given dataframe!')

    def _clip_the_dataframe(self):
        self.df = self.df.loc[self.first_day: self.last_day - timedelta(seconds=1), :]

    def _adjust_dataframe_indices(self):
        list_of_df_indices = self.df.index.tolist()
        list_of_moved_duplicates_index = []
        i = 0
        for idx, item in enumerate(list_of_df_indices):
            if (item.minute % self.time_frame) + item.second + item.microsecond > 0:
                offset = timedelta(minutes=(item.minute % self.time_frame), seconds=item.second,
                                   microseconds=item.microsecond)
                new_idx = item - offset
                if new_idx in list_of_df_indices:
                    list_of_df_indices[idx] = self.last_day + timedelta(seconds=idx + 1)
                    list_of_moved_duplicates_index.append(self.last_day + timedelta(seconds=idx + 1))
                else:
                    list_of_df_indices[idx] = new_idx
                i = i + 1
        self.df.index = list_of_df_indices
        self.number_of_candles_with_bad_index = i
        j = 0
        for idx in list_of_moved_duplicates_index:
            self.df.drop(idx, inplace=True)
            j = j + 1
        self.df.index.set_names('datetime', inplace=True)
        self.number_of_candles_with_two_index = j

    def _reset_first_candle(self):
        self.df.loc[self.first_day] = self.df.iloc[0, :]

    def _given_day_has_missing_data(self, date):
        return len(self.df.loc[date.strftime('%Y-%m-%d')]) < self.number_of_candles_in_each_day

    def _list_of_daily_candles(self, date):
        for i in range(self.number_of_candles_in_each_day):
            yield date + timedelta(minutes=i * self.time_frame)

    def _list_of_last_days_candles(self, date):
        last_date = datetime.strptime(self.last_day.date().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
        number_of_candles = int(np.floor((self.last_day - last_date).seconds / (60 * self.time_frame))) + 1
        for i in range(number_of_candles):
            yield date + timedelta(minutes=i * self.time_frame)

    def _impute_the_missing_candle(self, start_of_missing_candle):
        number_of_missing_candles, start_of_next_available_candle = \
            self._find_next_available_candle_and_number_of_missing_candles(start_of_missing_candle)
        if self.causal:
            self._causal_estimate_ohlcv_and_insert_the_candles(start_of_missing_candle, number_of_missing_candles)
        else:
            self._estimate_ohlcv_and_insert_the_candles(start_of_missing_candle, start_of_next_available_candle,
                                                        number_of_missing_candles)
        return number_of_missing_candles

    def _find_next_available_candle_and_number_of_missing_candles(self, current_index):
        i = 0
        t = current_index
        while t not in self.df.index:
            i += 1
            t += timedelta(minutes=self.time_frame)
            if t > self.last_day:
                raise Exception('Last candle was missing from the dataframe!')
        return i, t

    def _causal_estimate_ohlcv_and_insert_the_candles(self, missing_candle_idx, number_of_candles):
        for _ in range(number_of_candles):
            prev_index = missing_candle_idx - timedelta(minutes=self.time_frame)

            self.df.loc[missing_candle_idx] = self.df.loc[prev_index].to_dict()

            missing_candle_idx += timedelta(minutes=self.time_frame)
            number_of_candles += -1
            self._increase_number_of_missing_candles()

    def _estimate_ohlcv_and_insert_the_candles(self, missing_candle_idx, next_index, number_of_candles):
        for _ in range(number_of_candles):
            prev_index = missing_candle_idx - timedelta(minutes=self.time_frame)

            open_ = self.df.loc[prev_index].close
            high_ = (number_of_candles * self.df.loc[prev_index].high + self.df.loc[next_index].high) / (
                    number_of_candles + 1)
            low_ = (number_of_candles * self.df.loc[prev_index].low + self.df.loc[next_index].low) / (
                    number_of_candles + 1)
            close_ = ((number_of_candles - 1) * open_ + self.df.loc[next_index].open) / number_of_candles
            if 'volume' in self.df_columns:
                volume = (self.df.loc[prev_index].volume + self.df.loc[next_index].volume) / 2
                self.df.loc[missing_candle_idx] = {'open': open_, 'high': high_, 'low': low_, 'close': close_,
                                                   'volume': volume}
            else:
                self.df.loc[missing_candle_idx] = {'open': open_, 'high': high_, 'low': low_, 'close': close_}

            missing_candle_idx += timedelta(minutes=self.time_frame)
            number_of_candles += -1
            self._increase_number_of_missing_candles()

    def _increase_number_of_missing_candles(self):
        self.total_number_of_missing_candles += 1

    def _make_sure_no_where_volume_is_zero_or_negative(self):
        self.df.volume[self.df.volume < 1] = 1

    def return_complete_dataframe(self):
        return self.df


class ZeroOrderHold:
    def __init__(self, df: pd.DataFrame, time_frame_in_minutes=1440, first_day='2017-08-17 00:00:00',
                 last_day='2022-01-01 00:00:01'):
        self.df = df.copy()
        self.df_columns = self.df.columns.to_list()
        self.colprefix = self.df_columns[0].split('_')[0] + '_'
        self.time_frame = time_frame_in_minutes
        self.number_of_candles_in_each_day = int(1440 / time_frame_in_minutes)
        self.first_day = datetime.strptime(first_day, '%Y-%m-%d %H:%M:%S')
        self.last_day = datetime.strptime(last_day, '%Y-%m-%d %H:%M:%S')
        self.total_number_of_missing_candles = 0
        self.number_of_candles_with_bad_index = 0
        self.number_of_candles_with_two_index = 0

        self._run()

    def _run(self):
        self._assert_existence_of_the_first_and_last_candle_and_clip_the_dataframe()
        self._adjust_dataframe_indices()
        self._reset_first_candle()
        date = self.first_day
        while date < self.last_day:
            if self._given_day_has_missing_data(date):
                if date.date() == self.last_day.date():
                    gen = self._list_of_last_days_candles(date)
                else:
                    gen = self._list_of_daily_candles(date)
                for start_of_candle in gen:
                    if start_of_candle not in self.df.index:
                        self._impute_the_missing_candle(start_of_candle)
            date = date + timedelta(days=1)
        if self.colprefix + 'volume' in self.df_columns:
            self._make_sure_no_where_volume_is_zero_or_negative(self.colprefix + 'volume')
        self.df.sort_index(inplace=True)

    def _assert_existence_of_the_first_and_last_candle_and_clip_the_dataframe(self):
        timestamp_of_last_candle = self.last_day - timedelta(seconds=1)
        if timestamp_of_last_candle not in self.df.index.tolist():
            higher_points_df = self.df.loc[timestamp_of_last_candle:, :]
            if len(higher_points_df) > 0:
                self.df.loc[timestamp_of_last_candle] = higher_points_df.iloc[0, :].to_dict()
                self.df.sort_index(inplace=True)
            else:
                raise Exception('last candle must exist in the given dataframe!')
        if self.first_day not in self.df.index.tolist():
            lower_points_df = self.df.loc[:self.first_day, :]
            if len(lower_points_df) > 0:
                self.df.loc[self.first_day] = lower_points_df.iloc[-1, :].to_dict()
                self.df.sort_index(inplace=True)
            else:
                self.df.loc[self.first_day] = self.df.iloc[0]
                self.df.sort_index(inplace=True)
        self.df = self.df.loc[self.first_day:timestamp_of_last_candle, :]

    def _adjust_dataframe_indices(self):
        list_of_df_indices = self.df.index.tolist()
        list_of_moved_duplicates_index = []
        i = 0
        for idx, item in enumerate(list_of_df_indices):
            if (item.minute % self.time_frame) + item.second + item.microsecond > 0:
                offset = timedelta(minutes=(item.minute % self.time_frame), seconds=item.second,
                                   microseconds=item.microsecond)
                new_idx = item - offset
                if new_idx in list_of_df_indices:
                    list_of_df_indices[idx] = self.last_day + timedelta(seconds=idx + 1)
                    list_of_moved_duplicates_index.append(self.last_day + timedelta(seconds=idx + 1))
                else:
                    list_of_df_indices[idx] = new_idx
                i = i + 1
        self.df.index = list_of_df_indices
        self.number_of_candles_with_bad_index = i
        j = 0
        for idx in list_of_moved_duplicates_index:
            self.df.drop(idx, inplace=True)
            j = j + 1
        self.df.index.set_names('datetime', inplace=True)
        self.number_of_candles_with_two_index = j

    def _reset_first_candle(self):
        self.df.loc[self.first_day] = self.df.iloc[0, :]

    def _given_day_has_missing_data(self, date):
        return len(self.df.loc[date.strftime('%Y-%m-%d')]) < self.number_of_candles_in_each_day

    def _list_of_daily_candles(self, date):
        for i in range(self.number_of_candles_in_each_day):
            yield date + timedelta(minutes=i * self.time_frame)

    def _list_of_last_days_candles(self, date):
        last_date = datetime.strptime(self.last_day.date().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
        number_of_candles = int(np.floor((self.last_day - last_date).seconds / (60 * self.time_frame))) + 1
        for i in range(number_of_candles):
            yield date + timedelta(minutes=i * self.time_frame)

    def _impute_the_missing_candle(self, start_of_missing_candle):
        number_of_missing_candles, start_of_next_available_candle = \
            self._find_next_available_candle_and_number_of_missing_candles(start_of_missing_candle)

        self._repeat_data_from_the_previous_candle(start_of_missing_candle, number_of_missing_candles)

    def _find_next_available_candle_and_number_of_missing_candles(self, current_index):
        i = 0
        t = current_index
        while t not in self.df.index:
            i += 1
            t += timedelta(minutes=self.time_frame)
            if t > self.last_day:
                raise Exception('Last candle was missing from the dataframe!')
        return i, t

    def _repeat_data_from_the_previous_candle(self, missing_candle_idx, number_of_candles):
        for _ in range(number_of_candles):
            prev_index = missing_candle_idx - timedelta(minutes=self.time_frame)

            row_dict = self.df.loc[prev_index].to_dict()
            row_dict[self.colprefix + 'open'] = row_dict[self.colprefix + 'close']
            self.df.loc[missing_candle_idx] = row_dict

            missing_candle_idx += timedelta(minutes=self.time_frame)
            self._increase_recorded_number_of_missing_candles()

    def _increase_recorded_number_of_missing_candles(self):
        self.total_number_of_missing_candles += 1

    def _make_sure_no_where_volume_is_zero_or_negative(self, column_name):
        indexes = self.df[self.df[column_name] < 1].index
        self.df.loc[indexes, column_name] = 1

    def return_complete_dataframe(self):
        return self.df


class UpFirstDetector:
    def __init__(self, df, time_frame_in_minutes=1440, first_day='2017-08-17 00:00:00', last_day='2022-01-01 00:00:01'):
        self.df = df
        self.one_min_df = self._load_one_min_df()
        self.time_frame = time_frame_in_minutes
        self.first_candle = datetime.strptime(first_day, '%Y-%m-%d %H:%M:%S')
        self.last_candle = datetime.strptime(last_day, '%Y-%m-%d %H:%M:%S')
        self.run()

    def run(self):
        start_of_candle = self.first_candle
        up_first_column = []
        while start_of_candle < self.last_candle:
            if self._max_happened_before_min_in_this_candle(start_of_candle):
                up_first_column.append(True)
            else:
                up_first_column.append(False)
            start_of_candle += timedelta(minutes=self.time_frame)
        self.df['up_first'] = up_first_column

    @staticmethod
    def _load_one_min_df():
        df = pd.read_csv('PreprocessedData/BTCUSDT_1.csv')
        df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S")
        df.sort_values('datetime', inplace=True)
        df.set_index('datetime', inplace=True)
        return df

    def _max_happened_before_min_in_this_candle(self, start):
        end = start + timedelta(minutes=self.time_frame - 1)
        temp_df = self.one_min_df.loc[start.strftime('%Y-%m-%d %H:%M:%S'): end.strftime('%Y-%m-%d %H:%M:%S')]
        index_of_max = temp_df.high.idxmax()
        index_of_min = temp_df.low.idxmin()

        if index_of_max == index_of_min:
            return temp_df.loc[index_of_max, 'close'] <= temp_df.loc[index_of_max, 'open']
            # print('idx= ', index_of_max)
            # plt.figure()
            # plot_chart(temp_df)
            # plt.show()

        return index_of_max < index_of_min

    def return_new_df(self):
        return self.df


class TargetExtractor:
    def __init__(self, original_timeframe, timeframe):
        df = import_processed_data(f'PreprocessedData//BTCUSDT_{timeframe}.csv', index_column_name='datetime')
        df.rename({name: name[8:] for name in df.columns.to_list()}, axis=1, inplace=True)
        df.rename({'volatility_p_atr': 'patr'}, axis=1, inplace=True)

        self.df = df.loc[:, ['high', 'low', 'close', 'patr']].copy()
        self.number_of_candles_equivalent_to_one_original_candle = int(original_timeframe / timeframe)
        self.timeframe = timeframe
        self.targets_df = None

    def run(self):
        median_dt1 = self._smooth_out_the_price_series_with_kf(self.df.high.values, self.df.low.values)

        inverted_median_dt1 = self._smooth_out_the_price_series_with_kf(np.flipud(self.df.high.values),
                                                                        np.flipud(self.df.low.values))
        median_dt2 = np.flipud(inverted_median_dt1)

        self.targets_df = self._generate_the_targets_df(median_dt1, median_dt2)
        self.save_results()

    @staticmethod
    def _smooth_out_the_price_series_with_kf(high_values, low_values):
        tm = [[1, 1],
              [0, 1]]
        om = [1, 0]

        ism = [high_values[0], high_values[1] - high_values[0]]
        momentum_kf = KalmanFilter(transition_matrices=tm, observation_matrices=om, initial_state_mean=ism,
                                   em_vars=['transition_covariance', 'observation_covariance',
                                            'initial_state_covariance', 'transition_ofsets'])
        momentum_kf = momentum_kf.em(high_values[0:400], n_iter=30)
        estimated_highs = momentum_kf.smooth(high_values)[0]

        ism = [low_values[0], low_values[1] - low_values[0]]
        momentum_kf = KalmanFilter(transition_matrices=tm, observation_matrices=om, initial_state_mean=ism,
                                   em_vars=['transition_covariance', 'observation_covariance',
                                            'initial_state_covariance', 'transition_ofsets'])
        momentum_kf = momentum_kf.em(low_values[0:400], n_iter=30)
        estimated_lows = momentum_kf.smooth(low_values)[0]

        return np.array([(estimated_highs[:, 0] + estimated_lows[:, 0]) / 2]).T

    def _generate_the_targets_df(self, forward_median, backward_median):
        # targets1
        median_df1 = pd.DataFrame(data=(forward_median + backward_median) / 2, index=self.df.index,
                                  columns=['meds_mean'])

        median_df1['smoothed_change'] = (median_df1.meds_mean.shift(-1) - median_df1.meds_mean) / median_df1.meds_mean
        median_df1['target1'] = median_df1.smoothed_change >= 0
        median_df1['target1'] = median_df1['target1'].astype('int32')
        targets = pd.DataFrame(median_df1.target1.copy())

        # targets2
        targets['target2'] = self.remove_pullbacks(targets.copy())

        # targets3
        targets['target3'] = self.detect_reversals(self.df.copy())

        return targets

    @staticmethod
    def remove_pullbacks(df):
        df['target2'] = np.NAN
        target2_idx = df.columns.get_loc('target2')
        for i in range(len(df) - 4):
            temp_df = df.iloc[i: i + 5]
            df.iloc[i + 2, target2_idx] = TargetExtractor.smoothing(temp_df.target1)
        df.iloc[0:2, target2_idx] = df.iloc[0:2, df.columns.get_loc('target1')]
        df.iloc[-4:, target2_idx] = df.iloc[-4:, df.columns.get_loc('target1')]
        df['target2'] = df['target2'].astype('int32')
        return df.target2

    @staticmethod
    def smoothing(x):
        x1, x2, x3 = x[1], x[2], x[3]
        if (x[0] == x1) & (x3 == x[4]) & (x1 == x3) & (x2 != x3):
            return x1
        else:
            return x2

    @staticmethod
    def detect_reversals(df):
        df['target3'] = np.NAN
        target3_idx = df.columns.get_loc('target3')
        for i in range(len(df) - 32):
            temp_df = df.iloc[i: i + 33]
            df.iloc[i + 1, target3_idx] = TargetExtractor.reversal_detector(temp_df.close, temp_df.patr)

        list_of_last_candles = [i for i in range(len(df) - 32, len(df) - 5, 1)]
        for i, j in enumerate(list_of_last_candles):
            temp_df = df.iloc[j: j + 32 - i]
            df.iloc[j + 1, target3_idx] = TargetExtractor.reversal_detector(temp_df.close, temp_df.patr)
        df.fillna(value=33, inplace=True)
        df['target3'] = df['target3'].astype('int32')
        return df.target3

    @staticmethod
    def reversal_detector(x, atr):
        prev_price = x[0]
        price = x[1]
        patr = atr[1]

        l3atr = price < ((1 - 3 * patr) * x)
        s3atr = price > ((1 + 3 * patr) * x)
        l1atr = price < ((1 - 0.2 * patr) * x)
        s1atr = price > ((1 + 0.2 * patr) * x)

        if True in l3atr.values:
            if prev_price < price:
                return 0
            if True in s1atr.values:
                id_l3atr = l3atr.idxmax()
                id_satr = s1atr.idxmax()
                if id_l3atr <= id_satr:
                    return 1
            else:
                return 1
        if True in s3atr.values:
            if prev_price > price:
                return 0
            if True in l1atr.values:
                id_s3atr = s3atr.idxmax()
                id_latr = l1atr.idxmax()
                if id_s3atr <= id_latr:
                    return -1
            else:
                return -1
        return 0

    def _drop_the_formerly_repaired_nan_values_for_given_target(self, df, target):
        n = self.number_of_candles_equivalent_to_one_original_candle
        size = 70 if n == 1 else (71 * n) - 1
        df.drop(df.head(size).index, inplace=True)
        if target in ['target1', 'target2']:
            df.drop(df.tail(n).index, inplace=True)
        elif target == 'target3':
            df.drop(df.tail(5 * n).index, inplace=True)
        else:
            raise Exception('this target is not defiened')
        return df

    def save_results(self):
        target1_df = self._drop_the_formerly_repaired_nan_values_for_given_target(self.targets_df.copy(), 'target1')
        target1_df.drop(['target3'], axis=1, inplace=True)
        target1_df.to_csv(f'PreprocessedData//BTCUSDT_{self.timeframe}_targets1.csv')
        target3_df = self._drop_the_formerly_repaired_nan_values_for_given_target(self.targets_df.copy(), 'target3')
        target3_df.drop(['target1', 'target2'], axis=1, inplace=True)
        target3_df.to_csv(f'PreprocessedData//BTCUSDT_{self.timeframe}_targets3.csv')


class DataMixer:
    def __init__(self, df_name, timeframe, target, use_exogenous_data, first_day, last_day):
        self.df_name = df_name
        self.timeframe = timeframe
        self.use_exogenous_data = Exogenous[use_exogenous_data].value
        self.first_day = datetime.strptime(first_day, '%Y-%m-%d %H:%M:%S')
        self.last_day = datetime.strptime(last_day, '%Y-%m-%d %H:%M:%S')
        self.target = Target[target].value
        self.features_df = None
        self.targets_df = None

    def run(self):
        features_address, targets_address = self.get_address_of_features_and_targets()

        self.features_df = self.load_features(features_address)
        self.targets_df = self.load_targets(targets_address)

        self.save_the_result()

    def get_address_of_features_and_targets(self):
        if self.target in [Target.target1.value, Target.target2.value]:
            features_address = f'PreprocessedData//BTCUSDT_{self.timeframe}_1.csv'
            targets_address = f'PreprocessedData//BTCUSDT_{self.timeframe}_targets1.csv'
        else:
            features_address = f'PreprocessedData//BTCUSDT_{self.timeframe}_3.csv'
            targets_address = f'PreprocessedData//BTCUSDT_{self.timeframe}_targets3.csv'
        return features_address, targets_address

    def load_features(self, features_address):
        btc_df = import_processed_data(features_address, index_column_name='datetime')
        btc_df = btc_df.loc[self.first_day: self.last_day]
        btc_df = btc_df.fillna(method='bfill')

        if self.use_exogenous_data == Exogenous.regular.value:
            list_of_exo_dfs = []
            for name in ['BTCD', 'TOTAL2', 'USDTD', 'DXY', 'GOLD', 'SPX', 'UKOIL']:
                address = f'PreprocessedData//{name}_{self.timeframe}.csv'
                df = import_processed_data(address, index_column_name='datetime')
                list_of_exo_dfs.append(df)
            exo_data_df = pd.concat(list_of_exo_dfs, axis=1)
            exo_data_df = exo_data_df.loc[self.first_day: self.last_day]
            all_features_df = pd.concat([btc_df, exo_data_df], axis=1)
            return all_features_df.fillna(method='bfill')
        elif self.use_exogenous_data == Exogenous.none.value:
            return btc_df
        else:
            raise Exception('using this exogenous arg is not implemented yet!')

    def load_targets(self, targets_address):
        targets_df = import_processed_data(targets_address, index_column_name='datetime')
        return targets_df.loc[self.first_day:self.last_day]

    def save_the_result(self):
        self.features_df.to_csv(f'MixedData//{self.df_name}_features.csv')
        self.targets_df.to_csv(f'MixedData//{self.df_name}_targets.csv')


class FeatureEngineer:
    def __init__(self, df_name, model_name, target=Target.target1.name,
                 maximum_allowed_correlation_between_features=0.95):
        self.model_name = model_name
        self.target = target

        self.features_df, self.labels_df = get_features_and_labels(df_name)
        self.maximum_allowed_correlation = maximum_allowed_correlation_between_features
        self.initial_number_of_features = self.features_df.shape[1]
        self.list_of_selected_features = None
        self.number_of_features = None

    def run(self):
        my_feature_selector = self.FeatureSelector(self.features_df, self.labels_df, self.target,
                                                   self.maximum_allowed_correlation)
        self.list_of_selected_features = my_feature_selector.list_of_selected_features
        self.number_of_features = len(self.list_of_selected_features)
        self._write_feature_names_file()

    class FeatureSelector:
        def __init__(self, features_df, labels_df, target, maximum_allowed_correlation_between_features):
            self.features_df = features_df
            self.labels_df = labels_df
            self.target = target

            self.maximum_allowed_correlation_between_features = maximum_allowed_correlation_between_features
            self.selected_column_indexes = []

            self.list_of_mi = self._generate_list_of_mutual_information_between_features_and_labels()
            self.list_of_corr = self._generate_list_of_correlation_between_features_and_labels()
            self.features_correlation_matrix = self._generate_features_correlation_matrix()

            self._select_best_features()
            self.list_of_selected_features = self._create_list_of_selected_features()

        def _generate_list_of_mutual_information_between_features_and_labels(self):
            list_of_mi_change = mutual_info_classif(self.features_df.values, self.labels_df.loc[:, self.target])
            return list_of_mi_change

        def _generate_list_of_correlation_between_features_and_labels(self):
            correlation_measurement_df = pd.concat([self.features_df, self.labels_df], axis=1)
            correlation_matrix = correlation_measurement_df.corr()
            list_of_corr_change = list(np.abs(correlation_matrix.loc[self.target, :]).values)
            return list_of_corr_change

        def _generate_features_correlation_matrix(self):
            features_correlation_matrix = self.features_df.corr()
            return features_correlation_matrix

        def _select_best_features(self):
            for feature_index in range(self.features_correlation_matrix.shape[0]):
                indexes = self._get_indexes_of_features_that_have_high_correlation_with_selected_feature(feature_index)
                # if True in [idx in self.selected_column_indexes for idx in indexes]
                # pass
                # else:
                mi_list, corr_list, index_list = self._get_mi_and_corr_of_given_features_with_labels(indexes)
                self._append_the_best_feature_to_the_list_of_selected_features(mi_list, corr_list, index_list)

        def _get_indexes_of_features_that_have_high_correlation_with_selected_feature(self, index_of_feature):
            row_of_features_correlation = np.abs(self.features_correlation_matrix.iloc[index_of_feature, :])
            list_of_indexes = np.where(row_of_features_correlation > self.maximum_allowed_correlation_between_features)
            list_of_indexes = list(list_of_indexes)[0]
            return list_of_indexes

        def _get_mi_and_corr_of_given_features_with_labels(self, list_of_indexes):
            list_of_mi, list_of_corr, index_list = [], [], []

            for idx in list_of_indexes:
                mi, corr = self.list_of_mi[idx], self.list_of_corr[idx]
                list_of_mi.append(mi)
                list_of_corr.append(corr)
                index_list.append(idx)

            return list_of_mi, list_of_corr, index_list

        def _append_the_best_feature_to_the_list_of_selected_features(self, list_of_mi, list_of_corr, list_of_index):
            max_mi = np.amax(list_of_mi, axis=0)
            indexes_of_max_mi = list(np.where(list_of_mi == max_mi))[0]
            real_indexes_of_max_mi = [list_of_index[i] for i in indexes_of_max_mi]
            columns_check_list = [k in self.selected_column_indexes for k in real_indexes_of_max_mi]
            if True in columns_check_list:
                pass
            else:
                if len(indexes_of_max_mi) > 1:
                    corrs_of_max_mi = [list_of_corr[idx] for idx in indexes_of_max_mi]
                    arg_max_corr = np.argmax(corrs_of_max_mi)
                    self.selected_column_indexes.append(real_indexes_of_max_mi[arg_max_corr])
                else:
                    self.selected_column_indexes.append(list_of_index[np.argmax(list_of_mi)])

        def _create_list_of_selected_features(self):
            df_column_name = self.features_df.columns.to_list()
            final_columns = []
            for item in self.selected_column_indexes:
                if df_column_name[item] in final_columns:
                    pass
                else:
                    final_columns.append(df_column_name[item])
            return final_columns

    def _write_feature_names_file(self):
        selected_features = str(self.list_of_selected_features)
        with open(f'models_feature_names_files//features_of_{self.model_name}.txt', "w") as text_file:
            text_file.write(selected_features[1:-1])


class FeatureExtractor:
    def __init__(self, df_name):
        self.df_name = df_name
        self.all_features, self.targets = get_features_and_labels(df_name)

    def extract_features(self, model_name, save=True):
        list_of_features = read_features_list(f'models_feature_names_files//features_of_{model_name}.txt')
        selected_features_df = self.all_features.loc[:, list_of_features]
        if save is True:
            selected_features_df.to_csv(f'ProcessedData//{model_name}_ready_features.csv')
            self.targets.to_csv(f'ProcessedData//{model_name}_targets.csv')
        return selected_features_df, self.targets


class TrainTestValidationLoader:
    def __init__(self, model_data, target, training_portion=0.75, validation_portion=0.4, n_input_steps=21,
                 feature_scaler_range=(-1, 1), target_scaler_range=(-1, 1), original_time_frame=240):
        features_df, labels_df = self.get_features_and_labels(model_data)
        self.number_of_candles_equivalent_to_one_original_candle = int(original_time_frame * 60 / (
                features_df.index[1] - features_df.index[0]).total_seconds())
        self.targets = [target]
        self.full_df = self._mix_features_and_targets(features_df, labels_df)
        self.number_of_input_steps = n_input_steps
        self.feature_scaler = MinMaxScaler(feature_range=feature_scaler_range)
        self.target_scaler = MinMaxScaler(feature_range=target_scaler_range)
        self.x_shape = (-1, self.number_of_input_steps, len(features_df.columns.to_list()))

        self._run(training_portion, validation_portion)

    def _run(self, training_portion, validation_portion):
        df_train, df_test, df_val = self._train_test_val_split(training_portion, validation_portion,
                                                               self.number_of_candles_equivalent_to_one_original_candle)

        scaled_train_features, scaled_train_labels = self._scale(df_train)
        scaled_test_features, scaled_test_labels = self._scale(df_test, fit=False)
        scaled_val_features, scaled_val_labels = self._scale(df_val, fit=False)
        scaled_all_features, scaled_all_labels = self._scale(self.full_df, fit=False)

        self.x_train, self.y_train = self._reframe(scaled_train_features, scaled_train_labels)
        self.x_test, self.y_test = self._reframe(scaled_test_features, scaled_test_labels)
        self.x_val, self.y_val = self._reframe(scaled_val_features, scaled_val_labels)
        self.x_all, self.y_all = self._reframe(scaled_all_features, scaled_all_labels)

    @staticmethod
    def get_features_and_labels(model_data):
        if isinstance(model_data, list):
            features_df, labels_df = model_data[0], model_data[1]
        elif isinstance(model_data, str):
            features_address = f'ProcessedData//{model_data}_ready_features.csv'
            features_df = import_processed_data(features_address, index_column_name='datetime')
            labels_address = f'ProcessedData//{model_data}_targets.csv'
            labels_df = import_processed_data(labels_address, index_column_name='datetime')
        else:
            raise Exception('The given model data should either be a list of features and labels or'
                            ' a str, which is a valid model name!')
        return features_df, labels_df

    def _mix_features_and_targets(self, features_df, labels_df):
        df = features_df.copy()
        df[self.targets] = labels_df[self.targets]
        return df

    def _train_test_val_split(self, train_portion, validation_portion, n):
        if n == 1:
            size1 = int(len(self.full_df) * train_portion)
        else:
            size1 = (int((len(self.full_df) + n - 1) * train_portion / n) * n)
        df_train = self.full_df[0:size1].copy()
        df_test_and_val = self.full_df[size1 - (n * (self.number_of_input_steps - 1)):].copy()
        if n == 1:
            size2 = int(len(df_test_and_val) * (1 - validation_portion))
        else:
            size2 = (int((len(df_test_and_val) + n - 1) * (1 - validation_portion) / n) * n)
        df_test = df_test_and_val[0:size2].copy()
        df_val = df_test_and_val[size2 - (n * (self.number_of_input_steps - 1)):].copy()
        return df_train, df_test, df_val

    def _scale(self, df, fit=True):
        features = df.drop(columns=self.targets)
        targets = df.loc[:, self.targets]
        if fit:
            self.feature_scaler.fit(features.values)
            self.target_scaler.fit(targets.values)
        features.loc[:, :] = self.feature_scaler.transform(features.values)
        targets.loc[:, :] = self.target_scaler.transform(targets.values)
        return features, targets

    def _reframe(self, features, labels):
        stacked_features_df = pd.DataFrame()
        for i in range(self.number_of_input_steps - 1, -1, -1):
            cols = features.shift(i)
            rename_dict = {x: x + f'_minus_{i}' for x in list(features.columns)}
            cols = cols.rename(columns=rename_dict)
            stacked_features_df = pd.concat([stacked_features_df, cols], axis=1)
        n = self.number_of_candles_equivalent_to_one_original_candle
        stacked_features_df.drop(stacked_features_df.head((self.number_of_input_steps - 1) * n).index, inplace=True)

        labels_df = labels.drop(labels.head((self.number_of_input_steps - 1) * n).index)

        return stacked_features_df, labels_df

    def get_reframed_train_data(self):
        x = np.asarray(self.x_train.values.reshape(self.x_shape)).astype('float64')
        y = np.asarray(self.y_train.values.reshape(-1, 1, 1)).astype('float64')
        return x, y

    def get_reframed_test_data(self):
        x = np.asarray(self.x_test.values.reshape(self.x_shape)).astype('float64')
        y = np.asarray(self.y_test.values.reshape(-1, 1, 1)).astype('float64')
        return x, y

    def get_reframed_val_data(self):
        x = np.asarray(self.x_val.values.reshape(self.x_shape)).astype('float64')
        y = np.asarray(self.y_val.values.reshape(-1, 1, 1)).astype('float64')
        return x, y

    def get_reframed_all_data(self):
        x = np.asarray(self.x_all.values.reshape(self.x_shape)).astype('float64')
        y = np.asarray(self.y_all.values.reshape(-1, 1, 1)).astype('float64')
        return x, y

    def load_train_test_validation_data(self):
        return self.get_reframed_train_data(), self.get_reframed_test_data(), self.get_reframed_val_data()


class MixedDataTrainTestValidationLoader:
    def __init__(self, model_data, target, training_portion=0.75, validation_portion=0.4, n_input_steps=21,
                 feature_scaler_range=(-1, 1), target_scaler_range=(-1, 1), original_time_frame=240):
        features_df, labels_df = self.get_features_and_labels(model_data)
        self.x_train = []
        self.x_test = []
        self.x_val = []
        self.x_all = []

        groups = ['BTCUSDT', 'BTCD', 'TOTAL2', 'USDTD', 'DXY', 'GOLD', 'SPX', 'UKOIL']
        indexes = {}

        for feature_name in groups:
            temp = []
            for i, name in enumerate(features_df.columns.to_list()):
                if name.split('_')[0] == feature_name:
                    temp.append(i)
            indexes[feature_name] = temp

        self.number_of_candles_equivalent_to_one_original_candle = int(original_time_frame * 60 / (
                features_df.index[1] - features_df.index[0]).total_seconds())
        self.targets = [target]

        y_train, y_test, y_val, y_all = None, None, None, None
        for key in indexes.keys():
            selected_features_df = features_df.iloc[:, indexes[key]]
            self.full_df = self._mix_features_and_targets(selected_features_df, labels_df)
            self.number_of_input_steps = n_input_steps
            self.feature_scaler = MinMaxScaler(feature_range=feature_scaler_range)
            self.target_scaler = MinMaxScaler(feature_range=target_scaler_range)
            self.x_shape = (-1, self.number_of_input_steps, len(selected_features_df.columns.to_list()))

            x_train, y_train, x_test, y_test, x_val, y_val, x_all, y_all = \
                self._run(training_portion, validation_portion)
            self.x_train.append(np.asarray(x_train.values.reshape(self.x_shape)).astype('float64'))
            self.x_test.append(np.asarray(x_test.values.reshape(self.x_shape)).astype('float64'))
            self.x_val.append(np.asarray(x_val.values.reshape(self.x_shape)).astype('float64'))
            self.x_all.append(np.asarray(x_all.values.reshape(self.x_shape)).astype('float64'))

        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.y_all = y_all

    def _run(self, training_portion, validation_portion):
        df_train, df_test, df_val = self._train_test_val_split(training_portion, validation_portion,
                                                               self.number_of_candles_equivalent_to_one_original_candle)

        scaled_train_features, scaled_train_labels = self._scale(df_train)
        scaled_test_features, scaled_test_labels = self._scale(df_test, fit=False)
        scaled_val_features, scaled_val_labels = self._scale(df_val, fit=False)
        scaled_all_features, scaled_all_labels = self._scale(self.full_df, fit=False)

        x_train, y_train = self._reframe(scaled_train_features, scaled_train_labels)
        x_test, y_test = self._reframe(scaled_test_features, scaled_test_labels)
        x_val, y_val = self._reframe(scaled_val_features, scaled_val_labels)
        x_all, y_all = self._reframe(scaled_all_features, scaled_all_labels)

        return x_train, y_train, x_test, y_test, x_val, y_val, x_all, y_all

    @staticmethod
    def get_features_and_labels(model_data):
        if isinstance(model_data, list):
            features_df, labels_df = model_data[0], model_data[1]
        elif isinstance(model_data, str):
            features_address = f'ProcessedData//{model_data}_ready_features.csv'
            features_df = import_processed_data(features_address, index_column_name='datetime')
            labels_address = f'ProcessedData//{model_data}_targets.csv'
            labels_df = import_processed_data(labels_address, index_column_name='datetime')
        else:
            raise Exception('The given model data should either be a list of features and labels or'
                            ' a str, which is a valid model name!')
        return features_df, labels_df

    def _mix_features_and_targets(self, features_df, labels_df):
        df = features_df.copy()
        df[self.targets] = labels_df[self.targets]
        return df

    def _train_test_val_split(self, train_portion, validation_portion, n):
        if n == 1:
            size1 = int(len(self.full_df) * train_portion)
        else:
            size1 = (int((len(self.full_df) + n - 1) * train_portion / n) * n)
        df_train = self.full_df[0:size1].copy()
        df_test_and_val = self.full_df[size1 - (n * (self.number_of_input_steps - 1)):].copy()
        if n == 1:
            size2 = int(len(df_test_and_val) * (1 - validation_portion))
        else:
            size2 = (int((len(df_test_and_val) + n - 1) * (1 - validation_portion) / n) * n)
        df_test = df_test_and_val[0:size2].copy()
        df_val = df_test_and_val[size2 - (n * (self.number_of_input_steps - 1)):].copy()
        return df_train, df_test, df_val

    def _scale(self, df, fit=True):
        features = df.drop(columns=self.targets)
        targets = df.loc[:, self.targets]
        if fit:
            self.feature_scaler.fit(features.values)
            self.target_scaler.fit(targets.values)
        features.loc[:, :] = self.feature_scaler.transform(features.values)
        targets.loc[:, :] = self.target_scaler.transform(targets.values)
        return features, targets

    def _reframe(self, features, labels):
        stacked_features_df = pd.DataFrame()
        for i in range(self.number_of_input_steps - 1, -1, -1):
            cols = features.shift(i)
            rename_dict = {x: x + f'_minus_{i}' for x in list(features.columns)}
            cols = cols.rename(columns=rename_dict)
            stacked_features_df = pd.concat([stacked_features_df, cols], axis=1)
        n = self.number_of_candles_equivalent_to_one_original_candle
        stacked_features_df.drop(stacked_features_df.head((self.number_of_input_steps - 1) * n).index, inplace=True)

        labels_df = labels.drop(labels.head((self.number_of_input_steps - 1) * n).index)

        return stacked_features_df, labels_df

    def get_reframed_train_data(self):
        x = np.concatenate(self.x_train, axis=2)
        y = np.asarray(self.y_train.values.reshape(-1, 1, 1)).astype('float64')
        return x, y

    def get_reframed_test_data(self):
        x = np.concatenate(self.x_test, axis=2)
        y = np.asarray(self.y_test.values.reshape(-1, 1, 1)).astype('float64')
        return x, y

    def get_reframed_val_data(self):
        x = np.concatenate(self.x_val, axis=2)
        y = np.asarray(self.y_val.values.reshape(-1, 1, 1)).astype('float64')
        return x, y

    def get_reframed_all_data(self):
        x = np.concatenate(self.x_all, axis=2)
        y = np.asarray(self.y_all.values.reshape(-1, 1, 1)).astype('float64')
        return x, y

    def load_train_test_validation_data(self):
        return self.get_reframed_train_data(), self.get_reframed_test_data(), self.get_reframed_val_data()


def get_features_and_labels(df_name):
    features = import_processed_data('MixedData//' + df_name + '_features.csv', index_column_name='datetime')
    labels = import_processed_data('MixedData//' + df_name + '_targets.csv', index_column_name='datetime')
    return features, labels


def write_features_list(feature_names, file_name):
    selected_features = str(feature_names)
    with open(file_name, "w") as text_file:
        text_file.write(selected_features[1:-1])


def read_features_list(feature_names_file):
    list_of_features = []
    with open(feature_names_file, 'r+') as file:
        names = file.read()
        results_list = names.split(", ")
        for item in results_list:
            real_item = item[1:-1]
            if real_item not in list_of_features:
                list_of_features.append(real_item)
    return list_of_features


class Metrics(object):
    def __init__(self, y_true, predictions):
        self.y_true = y_true
        self.predictions = predictions
        self.residuals = self.y_true - self.predictions
        self.rmse = self.calculate_rmse(self.residuals)
        self.mae = self.calculate_mae(self.residuals)

    def calculate_rmse(self, residuals):
        """Root mean squared error."""
        return np.sqrt(np.mean(np.square(residuals)))

    def calculate_mae(self, residuals):
        """Mean absolute error."""
        return np.mean(np.abs(residuals))

    def calculate_malr(self, y_true, predictions):
        """Mean absolute log ratio."""
        return np.mean(np.abs(np.log(1 + predictions) - np.log(1 + y_true)))

    def report(self, name=None):
        if name is not None:
            print_string = '{} results'.format(name)
            print(print_string)
            print('~' * len(print_string))
            print('RMSE: {:2.3f}\nMAE: {:2.3f}'.format(self.rmse, self.mae))


class StrategyTester:
    def __init__(self, decisions_df):
        self.check_the_columns_of_df(decisions_df)
        self.length_of_df = self.check_the_length_of_df(decisions_df)
        self.df = decisions_df
        self.pnl_df = pd.DataFrame(index=self.df.index, columns=['pnl'])
        self.cpnl = [0]
        self.highest_cpnl = 0
        self.mdd = 0
        self.position = Position.Neutral.value
        self.position_price = None

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
        self.pnl_df = pd.DataFrame(index=self.df.index, columns=['pnl'])
        self.cpnl = [0]
        self.highest_cpnl = 0
        self.mdd = 0
        self.position = Position.Neutral.value
        self.position_price = None

    def _two_states_buy_only(self, generator):
        for date, prices, decision in generator:
            self._evaluate_the_mdd(prices)
            if decision == Decision.Long.value:
                self._long_signal(prices[-1], date)
            elif decision == Decision.Short.value and self.position == Position.Long.value:
                self._close_long_signal(prices[-1], date)
            else:
                pass
        self._if_there_is_any_open_position_close_it()

    def _four_states_buy_only(self, generator):
        for date, prices, decision in generator:
            self._evaluate_the_mdd(prices)
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
            self._evaluate_the_mdd(prices)
            if decision == Decision.Long.value:
                self._long_signal(prices[-1], date)
            elif decision == Decision.Short.value:
                self._short_signal(prices[-1], date)
            else:
                pass
        self._if_there_is_any_open_position_close_it()

    def _four_states_buy_and_sell(self, generator):
        for date, prices, decision in generator:
            self._evaluate_the_mdd(prices)
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
        self._close_current_position(price, date)

    def _close_current_position(self, price, date):
        if self.position == Position.Long.value:
            pnl = 100 * (price - self.position_price) / self.position_price
            self.pnl_df.loc[date, 'pnl'] = pnl
            self.cpnl.append(self.cpnl[-1] + pnl)
            self.highest_cpnl = np.maximum(self.highest_cpnl, self.cpnl[-1])
            self.position = Position.Neutral.value
        elif self.position == Position.Short.value:
            negative_pnl = 100 * (price - self.position_price) / self.position_price
            self.pnl_df.loc[date, 'pnl'] = -negative_pnl
            self.cpnl.append(self.cpnl[-1] - negative_pnl)
            self.highest_cpnl = np.maximum(self.highest_cpnl, self.cpnl[-1])
            self.position = Position.Neutral.value
        else:
            pass

    def _evaluate_the_mdd(self, prices):
        if self.position == Position.Neutral.value:
            pass
        else:
            if self.position == Position.Long.value:
                max_probable_loss = np.minimum((prices[1] - self.position_price) / self.position_price, 0)
            else:
                max_probable_loss = np.minimum(-1 * ((prices[0] - self.position_price) / self.position_price), 0)
            draw_down = np.maximum(self.highest_cpnl - (self.cpnl[-1] + max_probable_loss), 0)
            self.mdd = np.maximum(self.mdd, draw_down)
