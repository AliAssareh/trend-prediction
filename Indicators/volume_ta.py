import numpy as np
import pandas as pd
from Indicators.ta_utils import IndicatorMixin, _ema


class AccDistIndexIndicator(IndicatorMixin):
    """Accumulation/Distribution Index (ADI)

    Acting as leading indicator of price movements.

    https://school.stockcharts.com/doku.php?id=technical_indicators:accumulation_distribution_line

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, fillna: bool = False):
        self._high = high
        self._low = low
        self._close = close
        self._volume = volume
        self._fillna = fillna
        self._run()

    def _run(self):
        clv = ((self._close - self._low) - (self._high - self._close)) / (
                self._high - self._low)
        clv = clv.fillna(0.0)  # float division by zero
        adi = clv * self._volume
        self._adi = adi
        # self._adi = adi.cumsum()

    def acc_dist_index(self) -> pd.Series:
        """Accumulation/Distribution Index (ADI)

        Returns:
            pandas.Series: New feature generated.
        """
        adi = self._check_fillna(self._adi, value=0)
        return pd.Series(adi, name="adi")


class OnBalanceVolumeIndicator(IndicatorMixin):
    """On-balance volume (OBV)

    It relates price and volume in the stock market. OBV is based on a
    cumulative total volume.

    https://en.wikipedia.org/wiki/On-balance_volume

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, volume: pd.Series, fillna: bool = False):
        self._close = close
        self._volume = volume
        self._fillna = fillna
        self._run()

    def _run(self):
        obv = np.where(self._close < self._close.shift(1), -self._volume, self._volume)
        self._obv = pd.Series(obv, index=self._close.index)  # .cumsum()

    def on_balance_volume(self) -> pd.Series:
        """On-balance volume (OBV)

        Returns:
            pandas.Series: New feature generated.
        """
        obv = self._check_fillna(self._obv, value=0)
        return pd.Series(obv, name="obv")


class ChaikinMoneyFlowIndicator(IndicatorMixin):
    """Chaikin Money Flow (CMF)

    It measures the amount of Money Flow Volume over a specific period.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window=20, fillna=False):
        self._high = high
        self._low = low
        self._close = close
        self._volume = volume
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        mfv = ((self._close - self._low) - (self._high - self._close)) / (
                self._high - self._low
        )
        mfv = mfv.fillna(0.0)  # float division by zero
        mfv *= self._volume
        min_periods = 0 if self._fillna else self._window
        self._cmf = (
                mfv.rolling(self._window, min_periods=min_periods).sum()
                / self._volume.rolling(self._window, min_periods=min_periods).sum()
        )

    def chaikin_money_flow(self) -> pd.Series:
        """Chaikin Money Flow (CMF)

        Returns:
            pandas.Series: New feature generated.
        """
        cmf = self._check_fillna(self._cmf, value=0)
        return pd.Series(cmf, name="cmf")


class ForceIndexIndicator(IndicatorMixin):
    """Force Index (FI)

    It illustrates how strong the actual buying or selling pressure is. High
    positive values mean there is a strong rising trend, and low values signify
    a strong downward trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:force_index

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, volume: pd.Series, w1=13, w2=7, fillna: bool = False):
        self._close = close
        self._volume = volume
        self._window = w1
        self._window2 = w2
        self._fillna = fillna
        self._run()

    def _run(self):
        fi_series = (self._close - self._close.shift(1)) * self._volume / self._close.shift(1)
        self._fi = _ema(fi_series, self._window, fillna=self._fillna)
        self._fi_2 = _ema(fi_series, self._window2, fillna=self._fillna)

    def force_index(self) -> pd.Series:
        """Force Index (FI)

        Returns:
            pandas.Series: New feature generated.
        """
        fi_series = self._check_fillna(self._fi, value=0)
        return pd.Series(fi_series, name=f"fi_{self._window}")

    def force_index2(self) -> pd.Series:
        """Force Index (FI)

        Returns:
            pandas.Series: New feature generated.
        """
        fi_series = self._check_fillna(self._fi_2, value=0)
        return pd.Series(fi_series, name=f"fi_{self._window2}")


class MFIIndicator(IndicatorMixin):
    """Money Flow Index (MFI)

    Uses both price and volume to measure buying and selling pressure. It is
    positive when the typical price rises (buying pressure) and negative when
    the typical price declines (selling pressure). A ratio of positive and
    negative money flow is then plugged into an RSI formula to create an
    oscillator that moves between zero and one hundred.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:money_flow_index_mfi

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, volume, w1=14, w2=10, fillna=False):
        self._high = high
        self._low = low
        self._close = close
        self._volume = volume
        self._window = w1
        self._window2 = w2
        self._fillna = fillna
        self._run()

    def _run(self):
        typical_price = (self._high + self._low + self._close) / 3.0
        up_down = np.where(
            typical_price > typical_price.shift(1),
            1,
            np.where(typical_price < typical_price.shift(1), -1, 0),
        )
        mfr = typical_price * self._volume * up_down

        # Positive and negative money flow with n periods
        min_periods = 0 if self._fillna else self._window
        n_positive_mf = mfr.rolling(self._window, min_periods=min_periods).apply(
            lambda x: np.sum(np.where(x >= 0.0, x, 0.0)), raw=True
        )
        n_negative_mf = abs(
            mfr.rolling(self._window, min_periods=min_periods).apply(
                lambda x: np.sum(np.where(x < 0.0, x, 0.0)), raw=True
            )
        )

        mfi = n_positive_mf / n_negative_mf
        self._mfi = 100 - (100 / (1 + mfi))

        min_periods = 0 if self._fillna else self._window2
        n_positive_mf2 = mfr.rolling(self._window2, min_periods=min_periods).apply(
            lambda x: np.sum(np.where(x >= 0.0, x, 0.0)), raw=True
        )
        n_negative_mf2 = abs(
            mfr.rolling(self._window2, min_periods=min_periods).apply(
                lambda x: np.sum(np.where(x < 0.0, x, 0.0)), raw=True
            )
        )
        # Money flow index
        mfi2 = n_positive_mf2 / n_negative_mf2
        self._mfi2 = 100 - (100 / (1 + mfi2))

    def money_flow_index(self) -> pd.Series:
        """Money Flow Index (MFI)

        Returns:
            pandas.Series: New feature generated.
        """
        mfi = self._check_fillna(self._mfi, value=50)
        return pd.Series(mfi, name=f"mfi_{self._window}")

    def money_flow_index2(self) -> pd.Series:
        mfi2 = self._check_fillna(self._mfi2, value=50)
        return pd.Series(mfi2, name=f"mfi_{self._window2}")


class EaseOfMovementIndicator(IndicatorMixin):
    """Ease of movement (EoM, EMV)

    It relate an asset's price change to its volume and is particularly useful
    for assessing the strength of a trend.

    https://en.wikipedia.org/wiki/Ease_of_movement

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        volume(pandas.Series): dataset 'Volume' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, high: pd.Series, low: pd.Series, volume: pd.Series, window: int = 14, fillna: bool = False):
        self._high = high
        self._low = low
        self._volume = volume
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        prev_mid_price = (self._high.shift(1) + self._low.shift(1))
        numerator = (self._high.diff(1) + self._low.diff(1)) * (self._high - self._low)
        denominator = pd.Series(np.ones((len(self._high))), index=self._high.index) + (numerator / (self._low * prev_mid_price))
        self._emv = self._volume / denominator

    def ease_of_movement(self) -> pd.Series:
        """Ease of movement (EoM, EMV)

        Returns:
            pandas.Series: New feature generated.
        """
        emv = self._check_fillna(self._emv, value=0)
        return pd.Series(emv, name=f"eom_{self._window}")

    def sma_ease_of_movement(self) -> pd.Series:
        """Signal Ease of movement (EoM, EMV)

        Returns:
            pandas.Series: New feature generated.
        """
        min_periods = 0 if self._fillna else self._window
        emv = self._emv.rolling(self._window, min_periods=min_periods).mean()
        emv = self._check_fillna(emv, value=0)
        return pd.Series(emv, name=f"sma_eom_{self._window}")


class VolumePriceTrendIndicator(IndicatorMixin):
    """Volume-price trend (VPT)

    Is based on a running cumulative volume that adds or substracts a multiple
    of the percentage change in share price trend and current volume, depending
    upon the investment's upward or downward movements.

    https://en.wikipedia.org/wiki/Volume%E2%80%93price_trend

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, volume: pd.Series, fillna: bool = False):
        self._close = close
        self._volume = volume
        self._fillna = fillna
        self._run()

    def _run(self):
        vpt = self._volume * (
                (self._close - self._close.shift(1, fill_value=self._close.mean()))
                / self._close.shift(1, fill_value=self._close.mean())
        )
        self._vpt = vpt

    def volume_price_trend(self) -> pd.Series:
        """Volume-price trend (VPT)

        Returns:
            pandas.Series: New feature generated.
        """
        vpt = self._check_fillna(self._vpt, value=0)
        return pd.Series(vpt, name="vpt")


class NegativeVolumeIndexIndicator(IndicatorMixin):
    """Negative Volume Index (NVI)

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:negative_volume_inde

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        fillna(bool): if True, fill nan values with 1000.
    """

    def __init__(self, close: pd.Series, volume: pd.Series, fillna: bool = False):
        self._close = close
        self._volume = volume
        self._fillna = fillna
        self._run()

    def _run(self):
        price_change = self._close.pct_change()
        vol_decrease = self._volume.shift(1) > self._volume
        self._nvi = pd.Series(
            data=np.nan, index=self._close.index, dtype="float64", name="nvi"
        )
        self._nvi.iloc[0] = 0
        for i in range(1, len(self._nvi)):
            if vol_decrease.iloc[i]:
                self._nvi.iloc[i] = price_change.iloc[i]
            else:
                self._nvi.iloc[i] = self._nvi.iloc[i - 1]

    def negative_volume_index(self) -> pd.Series:
        """Negative Volume Index (NVI)

        Returns:
            pandas.Series: New feature generated.
        """
        # IDEA: There shouldn't be any na; might be better to throw exception
        nvi = self._check_fillna(self._nvi, value=0)
        return pd.Series(nvi, name="nvi")


class VolumeWeightedAveragePrice(IndicatorMixin):
    """Volume Weighted Average Price (VWAP)

    VWAP equals the dollar value of all trading periods divided
    by the total trading volume for the current day.
    The calculation starts when trading opens and ends when it closes.
    Because it is good for the current trading day only,
    intraday periods and data are used in the calculation.

    https://school.stockcharts.com/doku.php?id=technical_indicators:vwap_intraday

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """

    def __init__(
            self,
            opn: pd.Series,
            high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            volume: pd.Series,
            window: int = 14,
            fillna: bool = False,
    ):
        self._open = opn
        self._high = high
        self._low = low
        self._close = close
        self._volume = volume
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        # 1 typical price
        typical_price = (((self._high + self._low + self._close) / 3.0) - self._open) / self._open

        # 2 typical price * volume
        typical_price_volume = typical_price * self._volume

        # 3 total price * volume
        min_periods = 0 if self._fillna else self._window
        total_pv = typical_price_volume.rolling(
            self._window, min_periods=min_periods
        ).sum()

        # 4 total volume
        total_volume = self._volume.rolling(self._window, min_periods=min_periods).sum()

        self.vwap = total_pv / total_volume

    def volume_weighted_average_price(self) -> pd.Series:
        """Volume Weighted Average Price (VWAP)

        Returns:
            pandas.Series: New feature generated.
        """
        vwap = self._check_fillna(self.vwap)
        return pd.Series(vwap, name=f"vwap_{self._window}")
