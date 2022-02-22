from Indicators.stationary_indicators import add_all_stationary_ta_features
from FeatureEngineeringUtils.btc_feature_engineering_utils import import_raw_data, ZeroOrderHold
from datetime import datetime, timedelta
import pandas as pd


def recreate_the_spx_h1_candles_from_quarterly(name='SPX'):
    read_address = "Data//{}_{}.csv"
    write_address = "RepairedExoData//{}_{}.csv"
    m15_df = import_raw_data(read_address.format(name, 15), index_column_name='datetime')
    real_h1_df = import_raw_data(read_address.format(name, 60), index_column_name='datetime')
    reconstructor = ReconstructFrom15min(m15_df, first_day='2018-12-04 00:00:00')
    h1_df = reconstructor.h1_df
    date = datetime.strptime('2018-10-31 00:00:00', '%Y-%m-%d %H:%M:%S')
    start_of_m15_candles = datetime.strptime('2018-12-04 00:00:00', '%Y-%m-%d %H:%M:%S')
    while date < start_of_m15_candles:
        stamp = date + timedelta(hours=14)
        for i in range(7):
            if (stamp - timedelta(minutes=30)) in real_h1_df.index:
                h1_df.loc[stamp] = real_h1_df.loc[stamp - timedelta(minutes=30)]
            stamp = stamp + timedelta(hours=1)
        date = date + timedelta(days=1)
    h1_df.sort_index(inplace=True)
    h1_df.to_csv(write_address.format(name, '60'))


def recreate_the_h4_candles_from_hourly(name, first_day='2018-11-01 00:00:00', from_reconstructed_data=False):
    read_address = "RepairedExoData//{}_{}.csv" if from_reconstructed_data else "Data//{}_{}.csv"
    write_address = "RepairedExoData//{}_{}.csv"
    h1_df = import_raw_data(read_address.format(name, 60), index_column_name='datetime')
    reconstructor = ReconstructFromHourly(h1_df, first_day=first_day)
    h4_df = reconstructor.h4_df
    h4_df.to_csv(write_address.format(name, '240'))


def remove_the_volume_column(name, timeframe, reconstructed):
    address = "RepairedExoData//{}_{}.csv" if reconstructed else "Data//{}_{}.csv"
    df = import_raw_data(address.format(name, timeframe), index_column_name='datetime')
    if 'volume' in df.columns.to_list():
        df.drop(['volume'], axis=1, inplace=True)
        df.to_csv(address.format(name, timeframe))
        print(f'volume column >>removed<< form {address.format(name, timeframe)}')


def make_sure_no_row_has_volume_equal_to_zero(name, timeframe, reconstructed):
    address = "RepairedExoData//{}_{}.csv" if reconstructed else "Data//{}_{}.csv"
    df = import_raw_data(address.format(name, timeframe), index_column_name='datetime')
    if 'volume' in df.columns.to_list():
        indexes = df[df['volume'] < 1].index
        df.loc[indexes, 'volume'] = 1
        df.to_csv(address.format(name, timeframe))
        print(f'volume column modified for {address.format(name, timeframe)}')


class ExoPreprocessor:
    def __init__(self, name, timeframe, from_reconstructed_data=False, first_day='2019-01-01 00:00:00'):
        self.timeframe = int(timeframe)
        self.colprefix = name + '_'
        self.from_reconstructed_data = from_reconstructed_data
        self.first_day = first_day
        self.processed_df = None

    def run(self):
        if self.from_reconstructed_data:
            df = import_raw_data(f"RepairedExoData//{self.colprefix}{self.timeframe}.csv", index_column_name='datetime')
        else:
            df = import_raw_data(f"Data//{self.colprefix}{self.timeframe}.csv", index_column_name='datetime')

        alt_candles_df = self._create_alternative_candles(df)
        indicators_df = self._generate_indicators(df)
        features_df = self._create_features_df(alt_candles_df, indicators_df)
        self.processed_df = self.interpolate_the_data(features_df, self.timeframe, self.first_day)
        self.save_the_results()

    def _create_alternative_candles(self, df):
        temp_df = df.copy()
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
                        'volume': self.colprefix + 'volume'}, axis=1, inplace=True)

        temp_df.drop(columns=['previous_open', 'previous_close'], inplace=True)

        return temp_df

    def _generate_indicators(self, df):
        temp_df = df.copy()
        if 'volume' in temp_df.columns.to_list():
            indicators_df = add_all_stationary_ta_features(temp_df, 'open', 'high', 'low', 'close', 'volume',
                                                           colprefix=self.colprefix)
            indicators_df.drop(columns=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        else:
            indicators_df = add_all_stationary_ta_features(temp_df, 'open', 'high', 'low', 'close', 'None',
                                                           colprefix=self.colprefix)
            indicators_df.drop(columns=['open', 'high', 'low', 'close'], inplace=True)
        return indicators_df

    @staticmethod
    def _create_features_df(alt_candles_df, indicators_df):
        df = pd.concat([alt_candles_df, indicators_df], axis=1)
        return df

    @staticmethod
    def interpolate_the_data(df, timeframe, first_day):
        zoh = ZeroOrderHold(df, timeframe, first_day=first_day)
        interpolated_df = zoh.return_complete_dataframe()
        return interpolated_df

    def save_the_results(self):
        self.processed_df.to_csv(f'PreprocessedData//{self.colprefix}{self.timeframe}.csv')


class ReconstructFromHourly:
    def __init__(self, h1_df, first_day='2017-08-17 00:00:00', last_day='2022-01-01 00:00:01'):
        self.h1_df = h1_df.copy()
        self.h4_df = pd.DataFrame(columns=self.h1_df.columns.to_list())
        self.first_day = datetime.strptime(first_day, '%Y-%m-%d %H:%M:%S')
        self.last_day = datetime.strptime(last_day, '%Y-%m-%d %H:%M:%S')
        self._run()

    def _run(self):
        self._assert_existence_of_the_first_and_last_candle_and_clip_the_dataframe()
        date = self.first_day
        recreate_the_volume = 'volume' in self.h1_df.columns.to_list()
        while date < self.last_day:
            stamp = date
            for i in range(6):
                if stamp in self.h1_df.index:
                    self.h4_df.loc[stamp] = self._create_the_h4_candle(stamp, recreate_the_volume)
                stamp = stamp + timedelta(hours=4)
            date = date + timedelta(days=1)
        self.h4_df.sort_index(inplace=True)
        self.h4_df.index.name = 'datetime'

    def _assert_existence_of_the_first_and_last_candle_and_clip_the_dataframe(self):
        timestamp_of_last_candle = self.last_day - timedelta(seconds=1)
        if timestamp_of_last_candle not in self.h1_df.index.tolist():
            higher_points_df = self.h1_df.loc[timestamp_of_last_candle:]
            if len(higher_points_df) > 0:
                self.h1_df.loc[timestamp_of_last_candle] = higher_points_df.iloc[0].to_dict()
                self.h1_df.sort_index(inplace=True)
            else:
                raise Exception('last candle must exist in the given dataframe!')
        if self.first_day not in self.h1_df.index.tolist():
            lower_points_df = self.h1_df.loc[:self.first_day]
            if len(lower_points_df) > 0:
                self.h1_df.loc[self.first_day] = lower_points_df.iloc[-1].to_dict()
                self.h1_df.sort_index(inplace=True)
            else:
                self.h1_df.loc[self.first_day] = self.h1_df.iloc[0]
                self.h1_df.sort_index(inplace=True)
        self.h1_df = self.h1_df.loc[self.first_day:timestamp_of_last_candle]

    def _create_the_h4_candle(self, stamp, recreate_the_volume):
        h1_candles = self.h1_df.loc[stamp: stamp+timedelta(hours=3)]
        open_ = h1_candles.iloc[0, h1_candles.columns.get_loc('open')]
        high_ = h1_candles.high.max()
        low_ = h1_candles.low.min()
        close_ = h1_candles.iloc[-1, h1_candles.columns.get_loc('close')]
        if recreate_the_volume:
            volume_ = h1_candles.volume.sum()
            return {'open': open_, 'high': high_, 'low': low_, 'close': close_, 'volume': volume_}
        else:
            return {'open': open_, 'high': high_, 'low': low_, 'close': close_}


class ReconstructFrom15min:
    def __init__(self, m15_df, first_day='2017-08-17 00:00:00', last_day='2022-01-01 00:00:01'):
        self.m15_df = m15_df.copy()
        self.h1_df = pd.DataFrame(columns=self.m15_df.columns.to_list())
        self.first_day = datetime.strptime(first_day, '%Y-%m-%d %H:%M:%S')
        self.last_day = datetime.strptime(last_day, '%Y-%m-%d %H:%M:%S')
        self._run()

    def _run(self):
        self._assert_existence_of_the_first_and_last_candle_and_clip_the_dataframe()
        date = self.first_day
        while date < self.last_day:
            stamp = date
            for i in range(24):
                if stamp in self.m15_df.index:
                    self.h1_df.loc[stamp] = self._get_the_1h_candle(stamp)
                stamp = stamp + timedelta(hours=1)
            date = date + timedelta(days=1)
        self.h1_df.sort_index(inplace=True)
        self.h1_df.index.name = 'datetime'

    def _assert_existence_of_the_first_and_last_candle_and_clip_the_dataframe(self):
        timestamp_of_last_candle = self.last_day - timedelta(seconds=1)
        if timestamp_of_last_candle not in self.m15_df.index.tolist():
            higher_points_df = self.m15_df.loc[timestamp_of_last_candle:]
            if len(higher_points_df) > 0:
                self.m15_df.loc[timestamp_of_last_candle] = higher_points_df.iloc[0].to_dict()
                self.m15_df.sort_index(inplace=True)
            else:
                raise Exception('last candle must exist in the given dataframe!')
        if self.first_day not in self.m15_df.index.tolist():
            lower_points_df = self.m15_df.loc[:self.first_day]
            if len(lower_points_df) > 0:
                self.m15_df.loc[self.first_day] = lower_points_df.iloc[-1].to_dict()
                self.m15_df.sort_index(inplace=True)
            else:
                self.m15_df.loc[self.first_day] = self.m15_df.iloc[0]
                self.m15_df.sort_index(inplace=True)
        self.m15_df = self.m15_df.loc[self.first_day:timestamp_of_last_candle]

    def _get_the_1h_candle(self, stamp):
        m15_candles = self.m15_df.loc[stamp: stamp+timedelta(minutes=45)]
        open_ = m15_candles.iloc[0, m15_candles.columns.get_loc('open')]
        high_ = m15_candles.high.max()
        low_ = m15_candles.low.min()
        close_ = m15_candles.iloc[-1, m15_candles.columns.get_loc('close')]
        if 'volume' in m15_candles.columns:
            volume_ = m15_candles.volume.sum()
            return {'open': open_, 'high': high_, 'low': low_, 'close': close_, 'volume': volume_}
        else:
            return {'open': open_, 'high': high_, 'low': low_, 'close': close_}


class AdjustWeekends:
    def __init__(self, h4_df, h1_df, first_day='2017-08-17 00:00:00', last_day='2022-01-01 00:00:01'):
        self.h4_df = h4_df.copy()
        self.h1_df = h1_df.copy()
        self.first_day = datetime.strptime(first_day, '%Y-%m-%d %H:%M:%S')
        self.first_h1_date = self.h1_df.head(1).index.to_pydatetime()[0]
        self.last_day = datetime.strptime(last_day, '%Y-%m-%d %H:%M:%S')
        if self.last_day.date().weekday() == 6:  # sunday has a 22 candle that should go to 00
            self.h4_df.iloc[-1].index = self.last_day - timedelta(seconds=1)
        elif self.last_day.date().weekday() == 0:  # monday has a 2 candle that could go to 00
            self.h4_df.iloc[-6].index = self.last_day - timedelta(seconds=1)
        self._run()

    def _run(self):
        self._assert_existence_of_last_candle()
        self._clip_the_dataframe()
        self._reset_first_candle()
        date = self.first_day
        while date < self.last_day:
            if date.date().weekday() == 6:  # sunday
                if date.date() - timedelta(days=1) > self.first_h1_date.date():
                    self.h4_df.drop(self.h4_df.loc[date.strftime('%Y-%m-%d')].index, inplace=True)
                else:
                    next_day = date + timedelta(days=1)
                    self.h4_df.loc[next_day] = self.h4_df.loc[date + timedelta(hours=22)].copy()
                    self.h4_df.drop(self.h4_df.loc[date.strftime('%Y-%m-%d')].index, inplace=True)
            if date.date().weekday() == 0:  # monday
                if date.date() - timedelta(days=1) > self.first_h1_date.date():
                    self.h4_df.drop(self.h4_df.loc[date.strftime('%Y-%m-%d')].index, inplace=True)
                    stamp = date
                    for i in range(6):
                        self.h4_df.loc[stamp] = self._get_4h_candle(stamp)
                        stamp = stamp + timedelta(hours=4)
                else:
                    stamp = date + timedelta(hours=4)
                    for i in range(1, 6, 1):
                        self.h4_df.loc[stamp] = self.h4_df.loc[stamp - timedelta(hours=2)].copy()
                        self.h4_df.drop(self.h4_df.loc[stamp - timedelta(hours=2)].index, inplace=True)
                        stamp = stamp + timedelta(hours=4)
            date = date + timedelta(days=1)
        self.h4_df.sort_index(inplace=True)

    def _assert_existence_of_last_candle(self):
        if self.last_day - timedelta(seconds=1) not in self.h4_df.index.tolist():
            raise Exception('last candle must exist in the given dataframe!')

    def _clip_the_dataframe(self):
        self.h4_df = self.h4_df.loc[self.first_day: self.last_day - timedelta(seconds=1), :]

    def _reset_first_candle(self):
        self.h4_df.loc[self.first_day] = self.h4_df.iloc[0, :]

    def _get_4h_candle(self, stamp):
        h1_candles = self.h1_df.loc[stamp: stamp+timedelta(hours=3)]
        open_ = h1_candles.iloc[0, h1_candles.columns.get_loc('open')]
        high_ = h1_candles.high.max()
        low_ = h1_candles.low.min()
        close_ = h1_candles.iloc[-1, h1_candles.columns.get_loc('close')]
        if 'volume' in h1_candles.columns:
            volume_ = h1_candles.volume.sum()
            return {'open': open_, 'high': high_, 'low': low_, 'close': close_, 'volume': volume_}
        else:
            return {'open': open_, 'high': high_, 'low': low_, 'close': close_}