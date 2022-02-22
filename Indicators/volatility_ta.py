import numpy as np
import pandas as pd

from Indicators.ta_utils import IndicatorMixin


class AverageTrueRange(IndicatorMixin):
    """Average True Range (ATR)

    The indicator provide an indication of the degree of price volatility.
    Strong moves, in either direction, are often accompanied by large ranges,
    or large True Ranges.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
            self,
            high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            up_first=None,
            window: int = 14,
            fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._up_first = up_first
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        close_shift = self._close.shift(1)
        true_range = self._true_range(self._high, self._low, close_shift)

        atr = np.zeros(len(self._close))
        atr[self._window - 1] = true_range[0: self._window].mean()
        for i in range(self._window, len(atr)):
            atr[i] = (atr[i - 1] * (self._window - 1) + true_range.iloc[i]) / float(
                self._window
            )
        self._atr = pd.Series(data=atr, index=true_range.index)

        if self._up_first is not None:
            p_true_range = self.p_true_range(self._high, self._low, close_shift, self._up_first)
            p_atr = np.zeros(len(self._close))
            p_atr[self._window - 1] = p_true_range[0: self._window].mean()
            for i in range(self._window, len(p_atr)):
                p_atr[i] = (p_atr[i - 1] * (self._window - 1) + p_true_range.iloc[i]) / float(
                    self._window
                )
            for i in range(self._window - 1):
                p_atr[i] = np.NAN
            self.p_atr = pd.Series(data=p_atr, index=p_true_range.index)
            self.p_atr.fillna(self.p_atr.mean())
        else:
            self.p_atr = None

    def average_true_range(self) -> pd.Series:
        """Average True Range (ATR)

        Returns:
            pandas.Series: New feature generated.
        """
        atr = self._check_fillna(self._atr, value=0)
        return pd.Series(atr, name="atr")

    def p_average_true_range(self) -> pd.Series:
        """Average True Range (ATR)

        Returns:
            pandas.Series: New feature generated.
        """
        atr = self._check_fillna(self.p_atr, value=0)
        return pd.Series(atr, name="atr")


class BollingerBands(IndicatorMixin):
    """Bollinger Bands

    https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        window_dev(int): n factor standard deviation
        fillna(bool): if True, fill nan values.
    """

    def __init__(
            self,
            close: pd.Series,
            opn: pd.Series,
            window: int = 20,
            window_dev: int = 2,
            fillna: bool = False,
    ):
        self._close = close
        self._open = opn
        self._window = window
        self._window_dev = window_dev
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 0 if self._fillna else self._window
        self._mavg = self._close.rolling(self._window, min_periods=min_periods).mean()
        self._mstd = self._close.rolling(self._window, min_periods=min_periods).std(
            ddof=0
        )
        self._hband = self._mavg + self._window_dev * self._mstd
        self._lband = self._mavg - self._window_dev * self._mstd

    def bollinger_mavg(self) -> pd.Series:
        """Bollinger Channel Middle Band

        Returns:
            pandas.Series: New feature generated.
        """
        scaled_mavg = (self._mavg - self._open) / self._open
        mavg = self._check_fillna(scaled_mavg, value=-1)
        return pd.Series(mavg, name="mavg")

    def bollinger_hband(self) -> pd.Series:
        """Bollinger Channel High Band

        Returns:
            pandas.Series: New feature generated.
        """
        scaled_hband = (self._hband - self._open) / self._open
        hband = self._check_fillna(scaled_hband, value=-1)
        return pd.Series(hband, name="hband")

    def bollinger_lband(self) -> pd.Series:
        """Bollinger Channel Low Band

        Returns:
            pandas.Series: New feature generated.
        """
        scaled_lband = (self._lband - self._open) / self._open
        lband = self._check_fillna(scaled_lband, value=-1)
        return pd.Series(lband, name="lband")

    def bollinger_wband(self) -> pd.Series:
        """Bollinger Channel Band Width

        From: https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_band_width

        Returns:
            pandas.Series: New feature generated.
        """
        wband = ((self._hband - self._lband) / self._mavg) * 100
        wband = self._check_fillna(wband, value=0)
        return pd.Series(wband, name="bbiwband")

    def bollinger_pband(self) -> pd.Series:
        """Bollinger Channel Percentage Band

        From: https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_band_perce

        Returns:
            pandas.Series: New feature generated.
        """
        pband = (self._close - self._lband) / (self._hband - self._lband)
        pband = self._check_fillna(pband, value=0)
        return pd.Series(pband, name="bbipband")

    def bollinger_hband_indicator(self) -> pd.Series:
        """Bollinger Channel Indicator Crossing High Band (binary).

        It returns 1, if close is higher than bollinger_hband. Else, it returns 0.

        Returns:
            pandas.Series: New feature generated.
        """
        hband = pd.Series(
            np.where(self._close > self._hband, 1.0, 0.0), index=self._close.index
        )
        hband = self._check_fillna(hband, value=0)
        return pd.Series(hband, index=self._close.index, name="bbihband")

    def bollinger_lband_indicator(self) -> pd.Series:
        """Bollinger Channel Indicator Crossing Low Band (binary).

        It returns 1, if close is lower than bollinger_lband. Else, it returns 0.

        Returns:
            pandas.Series: New feature generated.
        """
        lband = pd.Series(
            np.where(self._close < self._lband, 1.0, 0.0), index=self._close.index
        )
        lband = self._check_fillna(lband, value=0)
        return pd.Series(lband, name="bbilband")


class KeltnerChannel(IndicatorMixin):
    """KeltnerChannel

    Keltner Channels are a trend following indicator used to identify reversals with channel breakouts and
    channel direction. Channels can also be used to identify overbought and oversold levels when the trend
    is flat.

    https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        window_atr(int): n atr period. Only valid if original_version param is False.
        fillna(bool): if True, fill nan values.
        original_version(bool): if True, use original version as the centerline (SMA of typical price)
            if False, use EMA of close as the centerline. More info:
            https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels
    """

    def __init__(
            self,
            high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            opn: pd.Series,
            up_first=None,
            window: int = 20,
            window_atr: int = 10,
            fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._open = opn
        self._up_first = up_first
        self._window = window
        self._window_atr = window_atr
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 1 if self._fillna else self._window
        self._tp = self._close.ewm(span=self._window, min_periods=min_periods, adjust=False).mean()
        self.p_tp = (self._tp - self._open) / self._open
        if self._up_first is None:
            ATR = AverageTrueRange(close=self._close, high=self._high, low=self._low,
                                   window=self._window_atr, fillna=self._fillna)
            _atr = ATR.average_true_range()
            self._tp_high = self._tp + (2 * _atr)
            self._tp_low = self._tp - (2 * _atr)
            self.p_tp_high = None
            self.p_tp_low = None
        else:
            ATR = AverageTrueRange(close=self._close, high=self._high, low=self._low, up_first=self._up_first,
                                   window=self._window_atr, fillna=self._fillna)
            p_atr = ATR.p_average_true_range()
            _atr = ATR.average_true_range()

            self._tp_high = self._tp + (2 * _atr)
            self._tp_low = self._tp - (2 * _atr)
            self.p_tp_high = self.p_tp + (2 * p_atr)
            self.p_tp_low = self.p_tp - (2 * p_atr)

    def keltner_channel_p_mband(self) -> pd.Series:
        """Keltner Channel Middle Band

        Returns:
            pandas.Series: New feature generated.
        """
        tp_middle = self._check_fillna(self.p_tp, value=-1)
        return pd.Series(tp_middle, name="mavg")

    def keltner_channel_hband(self) -> pd.Series:
        """Keltner Channel High Band

        Returns:
            pandas.Series: New feature generated.
        """
        tp_high = self._check_fillna(self._tp_high, value=-1)
        tp_high = (tp_high - self._open) / self._open
        return pd.Series(tp_high, name="kc_hband")

    def keltner_channel_lband(self) -> pd.Series:
        """Keltner Channel Low Band

        Returns:
            pandas.Series: New feature generated.
        """
        tp_low = self._check_fillna(self._tp_low, value=-1)
        tp_low = (tp_low - self._open) / self._open
        return pd.Series(tp_low, name="kc_lband")

    def keltner_channel_p_hband(self) -> pd.Series:
        """Keltner Channel High Band

        Returns:
            pandas.Series: New feature generated.
        """
        tp_high = self._check_fillna(self.p_tp_high, value=-1)
        return pd.Series(tp_high, name="kc_hband")

    def keltner_channel_p_lband(self) -> pd.Series:
        """Keltner Channel Low Band

        Returns:
            pandas.Series: New feature generated.
        """
        tp_low = self._check_fillna(self.p_tp_low, value=-1)
        return pd.Series(tp_low, name="kc_lband")

    def keltner_channel_wband(self) -> pd.Series:
        """Keltner Channel Band Width

        Returns:
            pandas.Series: New feature generated.
        """
        wband = ((self._tp_high - self._tp_low) / self._tp) * 100
        wband = self._check_fillna(wband, value=0)
        return pd.Series(wband, name="bbiwband")

    def keltner_channel_pband(self) -> pd.Series:
        """Keltner Channel Percentage Band

        Returns:
            pandas.Series: New feature generated.
        """
        pband = (self._close - self._tp_low) / (self._tp_high - self._tp_low)
        pband = self._check_fillna(pband, value=0)
        return pd.Series(pband, name="bbipband")

    def keltner_channel_hband_indicator(self) -> pd.Series:
        """Keltner Channel Indicator Crossing High Band (binary)

        It returns 1, if close is higher than keltner_channel_hband. Else, it returns 0.

        Returns:
            pandas.Series: New feature generated.
        """
        hband = pd.Series(
            np.where(self._close > self._tp_high, 1.0, 0.0), index=self._close.index
        )
        hband = self._check_fillna(hband, value=0)
        return pd.Series(hband, name="dcihband")

    def keltner_channel_lband_indicator(self) -> pd.Series:
        """Keltner Channel Indicator Crossing Low Band (binary)

        It returns 1, if close is lower than keltner_channel_lband. Else, it returns 0.

        Returns:
            pandas.Series: New feature generated.
        """
        lband = pd.Series(
            np.where(self._close < self._tp_low, 1.0, 0.0), index=self._close.index
        )
        lband = self._check_fillna(lband, value=0)
        return pd.Series(lband, name="dcilband")


class DonchianChannel(IndicatorMixin):
    """Donchian Channel

    https://www.investopedia.com/terms/d/donchianchannels.asp

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
            self,
            high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            opn: pd.Series,
            window: int = 20,
            offset: int = 0,
            fillna: bool = False,
    ):
        self._offset = offset
        self._close = close
        self._high = high
        self._low = low
        self._open = opn
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        self._min_periods = 1 if self._fillna else self._window
        self._hband = self._high.rolling(
            self._window, min_periods=self._min_periods
        ).max()
        self._lband = self._low.rolling(
            self._window, min_periods=self._min_periods
        ).min()

    def donchian_channel_hband(self) -> pd.Series:
        """Donchian Channel High Band

        Returns:
            pandas.Series: New feature generated.
        """
        hband = self._check_fillna(self._hband, value=-1)
        if self._offset != 0:
            hband = hband.shift(self._offset)
        hband = (hband - self._open) / self._open
        return pd.Series(hband, name="dchband")

    def donchian_channel_lband(self) -> pd.Series:
        """Donchian Channel Low Band

        Returns:
            pandas.Series: New feature generated.
        """
        lband = self._check_fillna(self._lband, value=-1)
        if self._offset != 0:
            lband = lband.shift(self._offset)
        lband = (lband - self._open) / self._open
        return pd.Series(lband, name="dclband")

    def donchian_channel_mband(self) -> pd.Series:
        """Donchian Channel Middle Band

        Returns:
            pandas.Series: New feature generated.
        """
        mband = (self._hband + self._lband) / 2.0
        mband = self._check_fillna(mband, value=-1)
        if self._offset != 0:
            mband = mband.shift(self._offset)
        mband = (mband - self._open) / self._open
        return pd.Series(mband, name="dcmband")

    def donchian_channel_wband(self) -> pd.Series:
        """Donchian Channel Band Width

        Returns:
            pandas.Series: New feature generated.
        """
        mavg = self._close.rolling(self._window, min_periods=self._min_periods).mean()
        wband = ((self._hband - self._lband) / mavg) * 100
        wband = self._check_fillna(wband, value=0)
        if self._offset != 0:
            wband = wband.shift(self._offset)
        return pd.Series(wband, name="dcwband")

    def donchian_channel_pband(self) -> pd.Series:
        """Donchian Channel Percentage Band

        Returns:
            pandas.Series: New feature generated.
        """
        pband = (self._close - self._lband) / (self._hband - self._lband)
        pband = self._check_fillna(pband, value=0)
        if self._offset != 0:
            pband = pband.shift(self._offset)
        return pd.Series(pband, name="dcpband")


class UlcerIndex(IndicatorMixin):
    """Ulcer Index

    https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ulcer_index

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, window: int = 14, fillna: bool = False):
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        _ui_max = self._close.rolling(self._window, min_periods=1).max()
        _r_i = 100 * (self._close - _ui_max) / _ui_max

        def ui_function():
            def _ui_function(x):
                return np.sqrt((x ** 2 / self._window).sum())

            return _ui_function

        self._ulcer_idx = _r_i.rolling(self._window).apply(ui_function(), raw=True)

        _ui_min = self._close.rolling(self._window, min_periods=1).min()
        n_r_i = 100 * (self._close - _ui_min) / _ui_min

        def Nui_function():
            def n_ui_function(x):
                return np.sqrt((x ** 2 / self._window).sum())

            return n_ui_function

        self.n_ulcer_idx = n_r_i.rolling(self._window).apply(Nui_function(), raw=True)

    def ulcer_index(self) -> pd.Series:
        """Ulcer Index (UI)

        Returns:
            pandas.Series: New feature generated.
        """
        ulcer_idx = self._check_fillna(self._ulcer_idx)
        return pd.Series(ulcer_idx, name="ui")

    def n_ulcer_index(self) -> pd.Series:
        """Ulcer Index (UI)

        Returns:
            pandas.Series: New feature generated.
        """
        ulcer_idx = self._check_fillna(self.n_ulcer_idx)
        return pd.Series(ulcer_idx, name="ui")
