import numpy as np
import pandas as pd

from Indicators.momentum_ta import (
    AwesomeOscillatorIndicator,
    KAMAIndicator,
    PercentagePriceOscillator,
    PercentageVolumeOscillator,
    ROCIndicator,
    RSIIndicator,
    StochasticOscillator,
    StochRSIIndicator,
    TSIIndicator,
    UltimateOscillator,
    WilliamsRIndicator,
)
from Indicators.trend_ta import (
    MACD,
    ADXIndicator,
    AroonIndicator,
    CCIIndicator,
    DPOIndicator,
    EMAIndicator,
    IchimokuIndicator,
    KSTIndicator,
    MassIndex,
    PSARIndicator,
    SMAIndicator,
    STCIndicator,
    TRIXIndicator,
    VortexIndicator,
    MinMaxIndicator,
)
from Indicators.volatility_ta import (
    AverageTrueRange,
    BollingerBands,
    DonchianChannel,
    KeltnerChannel,
    UlcerIndex,
)
from Indicators.volume_ta import (
    AccDistIndexIndicator,
    ChaikinMoneyFlowIndicator,
    EaseOfMovementIndicator,
    ForceIndexIndicator,
    MFIIndicator,
    NegativeVolumeIndexIndicator,
    OnBalanceVolumeIndicator,
    VolumePriceTrendIndicator,
    VolumeWeightedAveragePrice,
)


def online_diffs(df, df_columns, features_dict, colprefix=""):
    prev_open = df.open.shift(1)
    prev_close = df.close.shift(1)

    if f"{colprefix}volume" in df_columns:
        features_dict[f"{colprefix}volume"] = df.volume.iloc[-1]

    if f"{colprefix}up_first" in df_columns:
        features_dict[f"{colprefix}up_first"] = df.up_first.iloc[-1]

    if f"{colprefix}relative_change" in df_columns:
        relative_change = (df.open - prev_open) / prev_open
        features_dict[f"{colprefix}relative_change"] = relative_change.iloc[-1]

    if f"{colprefix}ho_percent" in df_columns:
        ho_percent = (df.high - df.open) / df.open
        features_dict[f"{colprefix}ho_percent"] = ho_percent.iloc[-1]

    if f"{colprefix}co_percent" in df_columns:
        co_percent = (df.close - df.open) / df.open
        features_dict[f"{colprefix}co_percent"] = co_percent.iloc[-1]

    if f"{colprefix}oc_percent" in df_columns:
        oc_percent = (df.open - prev_close) / prev_close
        features_dict[f"{colprefix}oc_percent"] = oc_percent.iloc[-1]

    if f"{colprefix}lo_percent" in df_columns:
        lo_percent = (df.low - df.open) / df.open
        features_dict[f"{colprefix}lo_percent"] = lo_percent.iloc[-1]

    return features_dict


def add_volume_ta(df: pd.DataFrame, opn: str, high: str, low: str, close: str, volume: str, fillna=False, colprefix=""):
    """Add volume technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        opn (str): Name of 'open' column.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    log_vol = df[volume].apply(np.log)

    # Accumulation Distribution Index
    df[f"{colprefix}volume_adi"] = AccDistIndexIndicator(
        high=df[high], low=df[low], close=df[close], volume=log_vol, fillna=fillna
    ).acc_dist_index()

    # On Balance Volume
    df[f"{colprefix}volume_obv"] = OnBalanceVolumeIndicator(
        close=df[close], volume=log_vol, fillna=fillna
    ).on_balance_volume()

    # Chaikin Money Flow
    df[f"{colprefix}volume_cmf"] = ChaikinMoneyFlowIndicator(
        high=df[high], low=df[low], close=df[close], volume=log_vol, fillna=fillna
    ).chaikin_money_flow()

    # Force Index
    indicator_fi = ForceIndexIndicator(close=df[close], volume=log_vol, w1=13, w2=7, fillna=fillna)
    df[f"{colprefix}volume_fi"] = indicator_fi.force_index()
    df[f"{colprefix}volume_fi_2"] = indicator_fi.force_index2()

    # Money Flow Indicator
    indicator_mfi = MFIIndicator(high=df[high], low=df[low], close=df[close], volume=log_vol, w1=14, w2=10,
                                 fillna=fillna)
    df[f"{colprefix}volume_mfi"] = indicator_mfi.money_flow_index()
    df[f"{colprefix}volume_mfi_2"] = indicator_mfi.money_flow_index2()

    # Ease of Movement
    indicator_eom = EaseOfMovementIndicator(high=df[high], low=df[low], volume=log_vol, window=14, fillna=fillna)
    df[f"{colprefix}volume_em"] = indicator_eom.ease_of_movement()
    df[f"{colprefix}volume_sma_em"] = indicator_eom.sma_ease_of_movement()

    # Volume Price Trend
    df[f"{colprefix}volume_vpt"] = VolumePriceTrendIndicator(
        close=df[close], volume=log_vol, fillna=fillna
    ).volume_price_trend()

    # Negative Volume Index
    df[f"{colprefix}volume_nvi"] = NegativeVolumeIndexIndicator(
        close=df[close], volume=log_vol, fillna=fillna
    ).negative_volume_index()

    # Volume Weighted Average Price
    df[f"{colprefix}volume_vwap"] = VolumeWeightedAveragePrice(
        opn=df[opn],
        high=df[high],
        low=df[low],
        close=df[close],
        volume=log_vol,
        window=14,
        fillna=fillna,
    ).volume_weighted_average_price()

    return df


def online_volume_ta(df: pd.DataFrame, df_columns, features_dict, colprefix=""):
    df1 = df.iloc[-72:, :]
    log_vol1 = df1.volume.apply(np.log)

    # Accumulation Distribution Index
    if f"{colprefix}volume_adi" in df_columns:
        adi = AccDistIndexIndicator(high=df1.high, low=df1.low, close=df1.close, volume=log_vol1).acc_dist_index()
        features_dict[f"{colprefix}volume_adi"] = adi.iloc[-1]

    # On Balance Volume
    if f"{colprefix}volume_obv" in df_columns:
        obv = OnBalanceVolumeIndicator(close=df1.close, volume=log_vol1).on_balance_volume()
        features_dict[f"{colprefix}volume_obv"] = obv.iloc[-1]

    # Chaikin Money Flow
    if f"{colprefix}volume_cmf" in df_columns:
        cmf = ChaikinMoneyFlowIndicator(high=df1.high, low=df1.low, close=df1.close, volume=log_vol1
        ).chaikin_money_flow()
        features_dict[f"{colprefix}volume_cmf"] = cmf.iloc[-1]

    # Force Index
    if (f"{colprefix}volume_fi" in df_columns) or (f"{colprefix}volume_fi_2" in df_columns):
        indicator_fi = ForceIndexIndicator(close=df1.close, volume=log_vol1, w1=13, w2=7)
        vfi1 = indicator_fi.force_index()
        vfi2 = indicator_fi.force_index2()
        if f"{colprefix}volume_fi" in df_columns:
            features_dict[f"{colprefix}volume_fi"] = vfi1.iloc[-1]
        if f"{colprefix}volume_fi_2" in df_columns:
            features_dict[f"{colprefix}volume_fi_2"] = vfi2.iloc[-1]

    # Money Flow Indicator
    if (f"{colprefix}volume_mfi" in df_columns) or (f"{colprefix}volume_mfi_2" in df_columns):
        indicator_mfi = MFIIndicator(high=df1.high, low=df1.low, close=df1.close, volume=log_vol1, w1=14, w2=10)
        mfi1 = indicator_mfi.money_flow_index()
        mfi2 = indicator_mfi.money_flow_index2()
        if f"{colprefix}volume_mfi" in df_columns:
            features_dict[f"{colprefix}volume_mfi"] = mfi1.iloc[-1]
        if f"{colprefix}volume_mfi_2" in df_columns:
            features_dict[f"{colprefix}volume_mfi_2"] = mfi2.iloc[-1]

    # Ease of Movement
    if (f"{colprefix}volume_em" in df_columns) or (f"{colprefix}volume_sma_em" in df_columns):
        indicator_eom = EaseOfMovementIndicator(high=df1.high, low=df1.low, volume=log_vol1, window=14)
        eom = indicator_eom.ease_of_movement()
        sma_eom = indicator_eom.sma_ease_of_movement()
        if f"{colprefix}volume_em" in df_columns:
            features_dict[f"{colprefix}volume_em"] = eom.iloc[-1]
        if f"{colprefix}volume_sma_em" in df_columns:
            features_dict[f"{colprefix}volume_sma_em"] = sma_eom.iloc[-1]

    # Volume Price Trend
    if f"{colprefix}volume_vpt" in df_columns:
        vpt = VolumePriceTrendIndicator(close=df1.close, volume=log_vol1).volume_price_trend()
        features_dict[f"{colprefix}volume_vpt"] = vpt.iloc[-1]

    # Negative Volume Index
    if f"{colprefix}volume_nvi" in df_columns:
        nvi = NegativeVolumeIndexIndicator(close=df1.close, volume=log_vol1).negative_volume_index()
        features_dict[f"{colprefix}volume_nvi"] = nvi.iloc[-1]

    # Volume Weighted Average Price
    if f"{colprefix}volume_vwap" in df_columns:
        vwap = VolumeWeightedAveragePrice(opn=df1.open, high=df1.high, low=df1.low, close=df1.close, volume=log_vol1,
                                          window=14).volume_weighted_average_price()
        features_dict[f"{colprefix}volume_vwap"] = vwap.index[-1]
    return features_dict


def add_volatility_ta(df, opn: str, high: str, low: str, close: str, up_first='None', fillna=False, colprefix=""):
    """Add volatility technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        opn (str): Name of 'open' column.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        up_first (str): Name of 'up_first' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    # Average True Range
    if up_first == 'None':
        df[f"{colprefix}volatility_atr"] = AverageTrueRange(
            close=df[close], high=df[high], low=df[low], window=10, fillna=fillna
        ).average_true_range()
    else:
        df[f"{colprefix}volatility_p_atr"] = AverageTrueRange(
            close=df[close], high=df[high], low=df[low], up_first=df[up_first], window=10, fillna=fillna
        ).p_average_true_range()

    # Bollinger Bands
    indicator_bb = BollingerBands(
        close=df[close], opn=df[opn], window=20, window_dev=2, fillna=fillna
    )
    df[f"{colprefix}volatility_bbm"] = indicator_bb.bollinger_mavg()
    df[f"{colprefix}volatility_bbh"] = indicator_bb.bollinger_hband()
    df[f"{colprefix}volatility_bbl"] = indicator_bb.bollinger_lband()
    df[f"{colprefix}volatility_bbw"] = indicator_bb.bollinger_wband()
    df[f"{colprefix}volatility_bbp"] = indicator_bb.bollinger_pband()
    df[f"{colprefix}volatility_bbhi"] = indicator_bb.bollinger_hband_indicator()
    df[f"{colprefix}volatility_bbli"] = indicator_bb.bollinger_lband_indicator()

    # Keltner Channel
    if up_first == 'None':
        indicator_kc = KeltnerChannel(
            close=df[close], high=df[high], low=df[low], opn=df[opn], window=10, fillna=fillna
        )
        # df[f"{colprefix}volatility_kcc"] = indicator_kc.keltner_channel_mband()
        df[f"{colprefix}volatility_kch"] = indicator_kc.keltner_channel_hband()
        df[f"{colprefix}volatility_kcl"] = indicator_kc.keltner_channel_lband()
        df[f"{colprefix}volatility_kcw"] = indicator_kc.keltner_channel_wband()
        df[f"{colprefix}volatility_kcp"] = indicator_kc.keltner_channel_pband()
        df[f"{colprefix}volatility_kchi"] = indicator_kc.keltner_channel_hband_indicator()
        df[f"{colprefix}volatility_kcli"] = indicator_kc.keltner_channel_lband_indicator()
    else:
        indicator_kc = KeltnerChannel(
            close=df[close], high=df[high], low=df[low], opn=df[opn], up_first=df[up_first], window=10, fillna=fillna
        )
        # df[f"{colprefix}volatility_kcc"] = indicator_kc.keltner_channel_mband()
        df[f"{colprefix}volatility_kch"] = indicator_kc.keltner_channel_p_hband()
        df[f"{colprefix}volatility_kcl"] = indicator_kc.keltner_channel_p_lband()
        df[f"{colprefix}volatility_kcw"] = indicator_kc.keltner_channel_wband()
        df[f"{colprefix}volatility_kcp"] = indicator_kc.keltner_channel_pband()
        df[f"{colprefix}volatility_kchi"] = indicator_kc.keltner_channel_hband_indicator()
        df[f"{colprefix}volatility_kcli"] = indicator_kc.keltner_channel_lband_indicator()

    # Donchian Channel
    indicator_dc = DonchianChannel(
        high=df[high], low=df[low], close=df[close], opn=df[opn], window=20, offset=0, fillna=fillna
    )
    df[f"{colprefix}volatility_dcl"] = indicator_dc.donchian_channel_lband()
    df[f"{colprefix}volatility_dch"] = indicator_dc.donchian_channel_hband()
    df[f"{colprefix}volatility_dcm"] = indicator_dc.donchian_channel_mband()
    df[f"{colprefix}volatility_dcw"] = indicator_dc.donchian_channel_wband()
    df[f"{colprefix}volatility_dcp"] = indicator_dc.donchian_channel_pband()

    # Ulcer Index
    ui = UlcerIndex(close=df[close], window=14, fillna=fillna)
    df[f"{colprefix}volatility_ui"] = ui.ulcer_index()
    df[f"{colprefix}volatility_nui"] = ui.n_ulcer_index()
    return df


def online_volatility_ta(df, df_columns, features_dict, colprefix=""):
    df1 = df.iloc[-72:, :]

    # Average True Range
    if f"{colprefix}volatility_p_atr" in df_columns:
        atr = AverageTrueRange(close=df1.close, high=df1.high, low=df1.low, up_first=df1.up_first,
                               window=10).p_average_true_range()
        features_dict[f"{colprefix}volatility_p_atr"] = atr.iloc[-1]

    # Bollinger Bands
    bb_list = [f"{colprefix}volatility_bbm", f"{colprefix}volatility_bbh", f"{colprefix}volatility_bbl",
               f"{colprefix}volatility_bbw", f"{colprefix}volatility_bbp", f"{colprefix}volatility_bbhi",
               f"{colprefix}volatility_bbli"]
    flag = []
    for item in bb_list:
        flag.append(True) if item in df_columns else flag.append(False)

    if True in flag:
        indicator_bb = BollingerBands(close=df1.close, opn=df1.open, window=20, window_dev=2)
        values_list = [indicator_bb.bollinger_mavg().iloc[-1], indicator_bb.bollinger_hband().iloc[-1],
                       indicator_bb.bollinger_lband().iloc[-1], indicator_bb.bollinger_wband().iloc[-1],
                       indicator_bb.bollinger_pband().iloc[-1], indicator_bb.bollinger_hband_indicator().iloc[-1],
                       indicator_bb.bollinger_lband_indicator().iloc[-1]]
        for i, f in enumerate(flag):
            if f is True:
                features_dict[bb_list[i]] = values_list[i]


    # Keltner Channel
    kc_list = [f"{colprefix}volatility_kch", f"{colprefix}volatility_kcl", f"{colprefix}volatility_kcw",
               f"{colprefix}volatility_kcp", f"{colprefix}volatility_kchi", f"{colprefix}volatility_kcli"]
    flag = []
    for item in kc_list:
        flag.append(True) if item in df_columns else flag.append(False)

    if True in flag:
        indicator_kc = KeltnerChannel(close=df1.close, high=df1.high, low=df1.low, opn=df1.open, up_first=df1.up_first,
                                      window=10)
        values_list = [indicator_kc.keltner_channel_p_hband().iloc[-1], indicator_kc.keltner_channel_p_lband().iloc[-1],
                       indicator_kc.keltner_channel_wband().iloc[-1], indicator_kc.keltner_channel_pband().iloc[-1],
                       indicator_kc.keltner_channel_hband_indicator().iloc[-1],
                       indicator_kc.keltner_channel_lband_indicator().iloc[-1]]
        for i, f in enumerate(flag):
            if f is True:
                features_dict[kc_list[i]] = values_list[i]

    # Donchian Channel
    dc_list = [f"{colprefix}volatility_dcl", f"{colprefix}volatility_dch", f"{colprefix}volatility_dcm",
               f"{colprefix}volatility_dcw", f"{colprefix}volatility_dcp"]
    flag = []
    for item in dc_list:
        flag.append(True) if item in df_columns else flag.append(False)

    if True in flag:
        indicator_dc = DonchianChannel(high=df1.high, low=df1.low, close=df1.close, opn=df1.open, window=20)
        values_list = [indicator_dc.donchian_channel_lband().iloc[-1], indicator_dc.donchian_channel_hband().iloc[-1],
                       indicator_dc.donchian_channel_mband().iloc[-1], indicator_dc.donchian_channel_wband().iloc[-1],
                       indicator_dc.donchian_channel_pband().iloc[-1]]
        for i, f in enumerate(flag):
            if f is True:
                features_dict[dc_list[i]] = values_list[i]

    # Ulcer Index
    if (f"{colprefix}volatility_ui" in df_columns) or (f"{colprefix}volatility_nui" in df_columns):
        ui = UlcerIndex(close=df1.close, window=14)
        if f"{colprefix}volatility_ui" in df_columns:
            features_dict[f"{colprefix}volatility_ui"] = ui.ulcer_index().iloc[-1]
        if f"{colprefix}volatility_nui" in df_columns:
            features_dict[f"{colprefix}volatility_nui"] = ui.n_ulcer_index().iloc[-1]
    return features_dict


def add_trend_ta(df, opn: str, high: str, low: str, close: str, fillna=False, colprefix: str = ""):
    """Add trend technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        opn (str): Name of 'open' column.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    # MACD
    indicator_macd = MACD(close=df[close], opn=df[opn], window_slow=26, window_fast=12, window_sign=9, fillna=fillna)
    df[f"{colprefix}trend_macd"] = indicator_macd.macd()
    df[f"{colprefix}trend_macd_signal"] = indicator_macd.macd_signal()
    # df[f"{colprefix}trend_macd_diff"] = indicator_macd.macd_diff()

    # SMAs
    df[f"{colprefix}trend_sma_12"] = SMAIndicator(close=df[close], opn=df[opn], window=12,
                                                  fillna=fillna).sma_indicator()
    df[f"{colprefix}trend_sma_26"] = SMAIndicator(close=df[close], opn=df[opn], window=26,
                                                  fillna=fillna).sma_indicator()
    df[f"{colprefix}trend_sma_50"] = SMAIndicator(close=df[close], opn=df[opn], window=50,
                                                  fillna=fillna).sma_indicator()

    # EMAs
    df[f"{colprefix}trend_ema_12"] = EMAIndicator(close=df[close], opn=df[opn], window=12,
                                                  fillna=fillna).ema_indicator()
    df[f"{colprefix}trend_ema_26"] = EMAIndicator(close=df[close], opn=df[opn], window=26,
                                                  fillna=fillna).ema_indicator()
    df[f"{colprefix}trend_ema_50"] = EMAIndicator(close=df[close], opn=df[opn], window=50,
                                                  fillna=fillna).ema_indicator()

    # Average Directional Movement Index (ADX)
    indicator_adx = ADXIndicator(
        high=df[high], low=df[low], close=df[close], window=14, fillna=fillna
    )
    df[f"{colprefix}trend_adx"] = indicator_adx.adx()
    df[f"{colprefix}trend_adx_pos"] = indicator_adx.adx_pos()
    df[f"{colprefix}trend_adx_neg"] = indicator_adx.adx_neg()

    # Vortex Indicator
    indicator_vortex = VortexIndicator(
        high=df[high], low=df[low], close=df[close], window=14, fillna=fillna
    )
    df[f"{colprefix}trend_vortex_ind_pos"] = indicator_vortex.vortex_indicator_pos()
    df[f"{colprefix}trend_vortex_ind_neg"] = indicator_vortex.vortex_indicator_neg()
    # df[f"{colprefix}trend_vortex_ind_diff"] = indicator_vortex.vortex_indicator_diff()

    # TRIX Indicator
    df[f"{colprefix}trend_trix"] = TRIXIndicator(
        close=df[close], window=15, fillna=fillna
    ).trix()

    # Mass Index
    df[f"{colprefix}trend_mass_index"] = MassIndex(
        high=df[high], low=df[low], window_fast=9, window_slow=25, fillna=fillna
    ).mass_index()

    # CCI Indicator
    df[f"{colprefix}trend_cci"] = CCIIndicator(
        high=df[high],
        low=df[low],
        close=df[close],
        window=20,
        constant=0.015,
        fillna=fillna,
    ).cci()

    # DPO Indicator
    df[f"{colprefix}trend_dpo"] = DPOIndicator(
        close=df[close], opn=df[opn], window=20, fillna=fillna
    ).dpo()

    # KST Indicator
    indicator_kst = KSTIndicator(
        close=df[close],
        roc1=10,
        roc2=15,
        roc3=20,
        roc4=30,
        window1=10,
        window2=10,
        window3=10,
        window4=15,
        nsig=9,
        fillna=fillna,
    )
    df[f"{colprefix}trend_kst"] = indicator_kst.kst()
    df[f"{colprefix}trend_kst_sig"] = indicator_kst.kst_sig()
    # df[f"{colprefix}trend_kst_diff"] = indicator_kst.kst_diff()

    # Ichimoku Indicator
    indicator_ichi = IchimokuIndicator(
        high=df[high],
        low=df[low],
        opn=df[opn],
        window1=9,
        window2=26,
        window3=52,
        fillna=fillna,
    )
    df[f"{colprefix}trend_ichimoku_conv"] = indicator_ichi.ichimoku_conversion_line()
    df[f"{colprefix}trend_ichimoku_base"] = indicator_ichi.ichimoku_base_line()
    df[f"{colprefix}trend_ichimoku_a"] = indicator_ichi.visual_ichimoku_a()
    df[f"{colprefix}trend_ichimoku_b"] = indicator_ichi.visual_ichimoku_b()

    # Aroon Indicator
    indicator_aroon = AroonIndicator(close=df[close], window=25, fillna=fillna)
    df[f"{colprefix}trend_aroon_up"] = indicator_aroon.aroon_up()
    df[f"{colprefix}trend_aroon_down"] = indicator_aroon.aroon_down()
    # df[f"{colprefix}trend_aroon_ind"] = indicator_aroon.aroon_indicator()

    indicator_aroon = AroonIndicator(close=df[high], window=25, fillna=fillna)
    df[f"{colprefix}trend_high_aroon_up"] = indicator_aroon.aroon_up()

    indicator_aroon = AroonIndicator(close=df[low], window=25, fillna=fillna)
    df[f"{colprefix}trend_low_aroon_down"] = indicator_aroon.aroon_down()

    # PSAR Indicator
    indicator_psar = PSARIndicator(
        high=df[high],
        low=df[low],
        close=df[close],
        opn=df[opn],
        step=0.02,
        max_step=0.20,
        fillna=fillna,
    )
    df[f'{colprefix}trend_psar'] = indicator_psar.psar()
    df[f'{colprefix}trend_psar_index'] = indicator_psar.psar_indicator()
    # df[f"{colprefix}trend_psar_up"] = indicator_psar.psar_up()
    # df[f"{colprefix}trend_psar_down"] = indicator_psar.psar_down()
    # df[f"{colprefix}trend_psar_up_indicator"] = indicator_psar.psar_up_indicator()
    # df[f"{colprefix}trend_psar_down_indicator"] = indicator_psar.psar_down_indicator()

    # Schaff Trend Cycle (STC)
    df[f"{colprefix}trend_stc"] = STCIndicator(
        close=df[close],
        window_slow=50,
        window_fast=23,
        cycle=10,
        smooth1=3,
        smooth2=3,
        fillna=fillna,
    ).stc()

    # MinMax cycle
    indicator_minmax = MinMaxIndicator(
        high=df[high],
        low=df[low],
        opn=df[opn],
    )
    df[f"{colprefix}trend_minmax_res1"] = indicator_minmax.res1()
    df[f"{colprefix}trend_minmax_res2"] = indicator_minmax.res2()
    df[f"{colprefix}trend_minmax_res3"] = indicator_minmax.res3()
    df[f"{colprefix}trend_minmax_a_res1"] = indicator_minmax.a_res1()
    df[f"{colprefix}trend_minmax_a_res2"] = indicator_minmax.a_res2()
    df[f"{colprefix}trend_minmax_a_res3"] = indicator_minmax.a_res3()
    df[f"{colprefix}trend_minmax_sup1"] = indicator_minmax.sup1()
    df[f"{colprefix}trend_minmax_sup2"] = indicator_minmax.sup2()
    df[f"{colprefix}trend_minmax_sup3"] = indicator_minmax.sup3()
    df[f"{colprefix}trend_minmax_a_sup1"] = indicator_minmax.a_sup1()
    df[f"{colprefix}trend_minmax_a_sup2"] = indicator_minmax.a_sup2()
    df[f"{colprefix}trend_minmax_a_sup3"] = indicator_minmax.a_sup3()

    return df


def online_trend_ta(df, df_columns, features_dict, colprefix: str = ""):
    df1 = df.iloc[-120:, :]

    # MACD
    macd_list = [f"{colprefix}trend_macd", f"{colprefix}trend_macd_signal", f"{colprefix}trend_macd_diff"]
    flag = []
    for item in macd_list:
        flag.append(True) if item in df_columns else flag.append(False)

    if True in flag:
        indicator_macd = MACD(close=df1.close, opn=df1.open, window_slow=26, window_fast=12, window_sign=9)
        values_list = [indicator_macd.macd().iloc[-1], indicator_macd.macd_signal().iloc[-1],
                       indicator_macd.macd_diff().iloc[-1]]
        for i, f in enumerate(flag):
            if f is True:
                features_dict[macd_list[i]] = values_list[i]

    # SMAs
    if f"{colprefix}trend_sma_12" in df_columns:
        features_dict[f"{colprefix}trend_sma_12"] = SMAIndicator(close=df1.close, opn=df1.open, window=12).sma_indicator().iloc[-1]
    if f"{colprefix}trend_sma_26" in df_columns:
        features_dict[f"{colprefix}trend_sma_26"] = SMAIndicator(close=df1.close, opn=df1.open, window=26).sma_indicator().iloc[-1]
    if f"{colprefix}trend_sma_50" in df_columns:
        features_dict[f"{colprefix}trend_sma_50"] = SMAIndicator(close=df1.close, opn=df1.open, window=50).sma_indicator().iloc[-1]

    # EMAs
    if f"{colprefix}trend_ema_12" in df_columns:
        features_dict[f"{colprefix}trend_ema_12"] = EMAIndicator(close=df1.close, opn=df1.open, window=12).ema_indicator().iloc[-1]
    if f"{colprefix}trend_ema_26" in df_columns:
        features_dict[f"{colprefix}trend_ema_26"] = EMAIndicator(close=df1.close, opn=df1.open, window=26).ema_indicator().iloc[-1]
    if f"{colprefix}trend_ema_50" in df_columns:
        features_dict[f"{colprefix}trend_ema_50"] = EMAIndicator(close=df1.close, opn=df1.open, window=50).ema_indicator().iloc[-1]

    # Average Directional Movement Index (ADX)
    adx_list = [f"{colprefix}trend_adx", f"{colprefix}trend_adx_pos", f"{colprefix}trend_adx_neg"]
    flag = []
    for item in adx_list:
        flag.append(True) if item in df_columns else flag.append(False)

    if True in flag:
        indicator_adx = ADXIndicator(high=df1.high, low=df1.low, close=df1.close, window=14)
        values_list = [indicator_adx.adx().iloc[-1], indicator_adx.adx_pos().iloc[-1], indicator_adx.adx_neg().iloc[-1]]
        for i, f in enumerate(flag):
            if f is True:
                features_dict[adx_list[i]] = values_list[i]

    # Vortex Indicator
    vtx_list = [f"{colprefix}trend_vortex_ind_pos", f"{colprefix}trend_vortex_ind_neg",
                f"{colprefix}trend_vortex_ind_diff"]
    flag = []
    for item in vtx_list:
        flag.append(True) if item in df_columns else flag.append(False)

    if True in flag:
        indicator_vortex = VortexIndicator(high=df1.high, low=df1.low, close=df1.close, window=14)
        values_list = [indicator_vortex.vortex_indicator_pos().iloc[-1],
                       indicator_vortex.vortex_indicator_neg().iloc[-1],
                       indicator_vortex.vortex_indicator_diff().iloc[-1]]
        for i, f in enumerate(flag):
            if f is True:
                features_dict[vtx_list[i]] = values_list[i]

    # TRIX Indicator
    if f"{colprefix}trend_trix" in df_columns:
        features_dict[f"{colprefix}trend_trix"] = TRIXIndicator(close=df1.close, window=15).trix().iloc[-1]

    # Mass Index
    if f"{colprefix}trend_mass_index" in df_columns:
        features_dict[f"{colprefix}trend_mass_index"] = MassIndex(high=df1.high, low=df1.low, window_fast=9, window_slow=25).mass_index().iloc[-1]

    # CCI Indicator
    if f"{colprefix}trend_cci" in df_columns:
        features_dict[f"{colprefix}trend_cci"] = CCIIndicator(high=df1.high, low=df1.low, close=df1.close, window=20, constant=0.015).cci().iloc[-1]

    # DPO Indicator
    if f"{colprefix}trend_dpo" in df_columns:
        features_dict[f"{colprefix}trend_dpo"] = DPOIndicator(close=df1.close, opn=df1.open, window=20).dpo().iloc[-1]

    # KST Indicator
    kst_list = [f"{colprefix}trend_kst", f"{colprefix}trend_kst_sig", f"{colprefix}trend_kst_diff"]
    flag = []
    for item in kst_list:
        flag.append(True) if item in df_columns else flag.append(False)

    if True in flag:
        indicator_kst = KSTIndicator(close=df1.close, roc1=10, roc2=15, roc3=20, roc4=30, window1=10, window2=10,
                                     window3=10, window4=15, nsig=9)
        values_list = [indicator_kst.kst().iloc[-1], indicator_kst.kst_sig().iloc[-1], indicator_kst.kst_diff().iloc[-1]]
        for i, f in enumerate(flag):
            if f is True:
                features_dict[kst_list[i]] = values_list[i]

    # Ichimoku Indicator
    ich_list = [f"{colprefix}trend_ichimoku_conv", f"{colprefix}trend_ichimoku_base", f"{colprefix}trend_ichimoku_a", f"{colprefix}trend_ichimoku_b"]
    flag = []
    for item in ich_list:
        flag.append(True) if item in df_columns else flag.append(False)

    if True in flag:
        indicator_ichi = IchimokuIndicator(high=df1.high, low=df1.low, opn=df1.open, window1=9, window2=26, window3=52)
        values_list = [indicator_ichi.ichimoku_conversion_line().iloc[-1], indicator_ichi.ichimoku_base_line().iloc[-1],
                       indicator_ichi.visual_ichimoku_a().iloc[-1], indicator_ichi.visual_ichimoku_b().iloc[-1]]
        for i, f in enumerate(flag):
            if f is True:
                features_dict[ich_list[i]] = values_list[i]

    # Aroon Indicator
    arn_list = [f"{colprefix}trend_aroon_up", f"{colprefix}trend_aroon_down", f"{colprefix}trend_aroon_ind"]
    flag = []
    for item in arn_list:
        flag.append(True) if item in df_columns else flag.append(False)

    if True in flag:
        indicator_aroon = AroonIndicator(close=df1.close, window=25)
        values_list = [indicator_aroon.aroon_up().iloc[-1], indicator_aroon.aroon_down().iloc[-1], indicator_aroon.aroon_indicator().iloc[-1]]
        for i, f in enumerate(flag):
            if f is True:
                features_dict[arn_list[i]] = values_list[i]

    if f"{colprefix}trend_high_aroon_up" in df_columns:
        indicator_aroon = AroonIndicator(close=df1.high, window=25)
        features_dict[f"{colprefix}trend_high_aroon_up"] = indicator_aroon.aroon_up().iloc[-1]

    if f"{colprefix}trend_low_aroon_down" in df_columns:
        indicator_aroon = AroonIndicator(close=df1.low, window=25)
        features_dict[f"{colprefix}trend_low_aroon_down"] = indicator_aroon.aroon_down()

    # PSAR Indicator
    psar_list = [f'{colprefix}trend_psar', f'{colprefix}trend_psar_index', f"{colprefix}trend_psar_up", f"{colprefix}trend_psar_down", f"{colprefix}trend_psar_up_indicator", f"{colprefix}trend_psar_down_indicator"]
    flag = []
    for item in psar_list:
        flag.append(True) if item in df_columns else flag.append(False)

    if True in flag:
        indicator_psar = PSARIndicator(high=df.high, low=df.low, close=df.close, opn=df.open, step=0.02, max_step=0.20)
        values_list = [indicator_psar.psar().iloc[-1], indicator_psar.psar_indicator().iloc[-1],
                       indicator_psar.psar_up().iloc[-1], indicator_psar.psar_down().iloc[-1],
                       indicator_psar.psar_up_indicator().iloc[-1], indicator_psar.psar_down_indicator().iloc[-1]]
        for i, f in enumerate(flag):
            if f is True:
                features_dict[psar_list[i]] = values_list[i]


    # Schaff Trend Cycle (STC)
    if f"{colprefix}trend_stc" in df_columns:
        features_dict[f"{colprefix}trend_stc"] = STCIndicator(close=df.close,window_slow=50,window_fast=23,cycle=10,
                                                              smooth1=3,smooth2=3).stc().iloc[-1]

    # MinMax cycle
    mmc_list = [f"{colprefix}trend_minmax_res1", f"{colprefix}trend_minmax_res2", f"{colprefix}trend_minmax_res3",
                f"{colprefix}trend_minmax_a_res1", f"{colprefix}trend_minmax_a_res2", f"{colprefix}trend_minmax_a_res3",
                f"{colprefix}trend_minmax_sup1", f"{colprefix}trend_minmax_sup2", f"{colprefix}trend_minmax_sup3",
                f"{colprefix}trend_minmax_a_sup1", f"{colprefix}trend_minmax_a_sup2", f"{colprefix}trend_minmax_a_sup3"]
    flag = []
    for item in mmc_list:
        flag.append(True) if item in df_columns else flag.append(False)

    if True in flag:
        df2 = df1.iloc[-53:, :]
        indicator_minmax = MinMaxIndicator(high=df2.high, low=df2.low, opn=df2.open)
        values_list = [indicator_minmax.res1().iloc[-1], indicator_minmax.res2().iloc[-1],
                       indicator_minmax.res3().iloc[-1], indicator_minmax.a_res1().iloc[-1],
                       indicator_minmax.a_res2().iloc[-1], indicator_minmax.a_res3().iloc[-1],
                       indicator_minmax.sup1().iloc[-1], indicator_minmax.sup2().iloc[-1],
                       indicator_minmax.sup3().iloc[-1], indicator_minmax.a_sup1().iloc[-1],
                       indicator_minmax.a_sup2().iloc[-1], indicator_minmax.a_sup3().iloc[-1]]
        for i, f in enumerate(flag):
            if f is True:
                features_dict[mmc_list[i]] = values_list[i]
    return features_dict


def add_momentum_ta(df, opn: str, high: str, low: str, close: str, volume='None', fillna=False, colprefix: str = ""):
    """Add trend technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        opn (str): Name of 'open' column.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    # Relative Strength Index (RSI)
    df[f"{colprefix}momentum_rsi"] = RSIIndicator(
        close=df[close], window=14, fillna=fillna
    ).rsi()

    # Stoch RSI (StochRSI)
    indicator_srsi = StochRSIIndicator(
        close=df[close], window=14, smooth1=3, smooth2=3, fillna=fillna
    )
    df[f"{colprefix}momentum_stoch_rsi"] = indicator_srsi.stochrsi()
    # df[f"{colprefix}momentum_stoch_rsi_k"] = indicator_srsi.stochrsi_k()
    # df[f"{colprefix}momentum_stoch_rsi_d"] = indicator_srsi.stochrsi_d()

    # TSI Indicator
    df[f"{colprefix}momentum_tsi"] = TSIIndicator(
        close=df[close], window_slow=25, window_fast=13, fillna=fillna
    ).tsi()

    # Ultimate Oscillator
    df[f"{colprefix}momentum_uo"] = UltimateOscillator(
        high=df[high],
        low=df[low],
        close=df[close],
        window1=7,
        window2=14,
        window3=28,
        weight1=4.0,
        weight2=2.0,
        weight3=1.0,
        fillna=fillna,
    ).ultimate_oscillator()

    # Stoch Indicator
    indicator_so = StochasticOscillator(
        high=df[high],
        low=df[low],
        close=df[close],
        window=14,
        smooth_window=3,
        fillna=fillna,
    )
    df[f"{colprefix}momentum_stoch"] = indicator_so.stoch()
    df[f"{colprefix}momentum_stoch_signal"] = indicator_so.stoch_signal()

    # Williams R Indicator
    # df[f"{colprefix}momentum_wr"] = WilliamsRIndicator(
    #     high=df[high], low=df[low], close=df[close], lbp=14, fillna=fillna
    # ).williams_r()
    # WR = WilliamsRIndicator(high=df[high], low=df[low], close=df[close], lbp=14, fillna=fillna)

    # Awesome Oscillator
    df[f"{colprefix}momentum_ao"] = AwesomeOscillatorIndicator(
        high=df[high], low=df[low], opn=df[opn], window1=5, window2=34, fillna=fillna
    ).awesome_oscillator()

    # KAMA
    df[f"{colprefix}momentum_kama"] = KAMAIndicator(
        close=df[close], opn=df[opn], window=10, pow1=2, pow2=30, fillna=fillna
    ).kama()

    # Rate Of Change
    df[f"{colprefix}momentum_roc"] = ROCIndicator(
        close=df[close], window=12, fillna=fillna
    ).roc()

    # Percentage Price Oscillator
    indicator_ppo = PercentagePriceOscillator(
        close=df[close], window_slow=26, window_fast=12, window_sign=9, fillna=fillna
    )
    df[f"{colprefix}momentum_ppo"] = indicator_ppo.ppo()
    df[f"{colprefix}momentum_ppo_signal"] = indicator_ppo.ppo_signal()
    # df[f"{colprefix}momentum_ppo_hist"] = indicator_ppo.ppo_hist()

    # Percentage Volume Oscillator
    if volume != 'None':
        indicator_pvo = PercentageVolumeOscillator(
            volume=df[volume].apply(np.log), window_slow=26, window_fast=12, window_sign=9, fillna=fillna
        )
        df[f"{colprefix}momentum_pvo"] = indicator_pvo.pvo()
        df[f"{colprefix}momentum_pvo_signal"] = indicator_pvo.pvo_signal()
        # df[f"{colprefix}momentum_pvo_hist"] = indicator_pvo.pvo_hist()

    return df


def online_momentum_ta(df, df_columns, features_dict, colprefix=""):
    df1 = df.iloc[-72:, :]
    log_vol1 = df1.volume.apply(np.log)

    # Relative Strength Index (RSI)
    if f"{colprefix}momentum_rsi" in df_columns:
        features_dict[f"{colprefix}momentum_rsi"] = RSIIndicator(close=df1.close, window=14).rsi().iloc[-1]

    # Stoch RSI (StochRSI)
    srsi_list = [f"{colprefix}momentum_stoch_rsi", f"{colprefix}momentum_stoch_rsi_k",
                 f"{colprefix}momentum_stoch_rsi_d"]
    flag = []
    for item in srsi_list:
        flag.append(True) if item in df_columns else flag.append(False)

    if True in flag:
        indicator_srsi = StochRSIIndicator(close=df1.close, window=14, smooth1=3, smooth2=3)
        values_list = [indicator_srsi.stochrsi().iloc[-1], indicator_srsi.stochrsi_k().iloc[-1], indicator_srsi.stochrsi_d().iloc[-1]]
        for i, f in enumerate(flag):
            if f is True:
                features_dict[srsi_list[i]] = values_list[i]

    # TSI Indicator
    if f"{colprefix}momentum_tsi" in df_columns:
        features_dict[f"{colprefix}momentum_tsi"] = TSIIndicator(close=df1.close, window_slow=25, window_fast=13).tsi().iloc[-1]

    # Ultimate Oscillator
    if f"{colprefix}momentum_uo" in df_columns:
        features_dict[f"{colprefix}momentum_uo"] = UltimateOscillator(high=df1.high,low=df1.low,close=df1.close,window1=7,
            window2=14,window3=28,weight1=4.0,weight2=2.0,weight3=1.0,).ultimate_oscillator().iloc[-1]

    # Stoch Indicator
    if (f"{colprefix}momentum_stoch" in df_columns) or (f"{colprefix}momentum_stoch_signal" in df_columns):
        indicator_so = StochasticOscillator(high=df1.high, low=df1.low, close=df1.close, window=14, smooth_window=3)
        if f"{colprefix}momentum_stoch" in df_columns:
            features_dict[f"{colprefix}momentum_stoch"] = indicator_so.stoch().iloc[-1]
        if f"{colprefix}momentum_stoch_signal" in df_columns:
            features_dict[f"{colprefix}momentum_stoch_signal"] = indicator_so.stoch_signal().iloc[-1]

    # Williams R Indicator
    # df[f"{colprefix}momentum_wr"] = WilliamsRIndicator(
    #     high=df[high], low=df[low], close=df[close], lbp=14, fillna=fillna
    # ).williams_r()
    # WR = WilliamsRIndicator(high=df[high], low=df[low], close=df[close], lbp=14, fillna=fillna)

    # Awesome Oscillator
    if f"{colprefix}momentum_ao" in df_columns:
        features_dict[f"{colprefix}momentum_ao"] = AwesomeOscillatorIndicator(
            high=df1.high, low=df1.low, opn=df1.open, window1=5, window2=34).awesome_oscillator().iloc[-1]

    # KAMA
    if f"{colprefix}momentum_kama" in df_columns:
        features_dict[f"{colprefix}momentum_kama"] = KAMAIndicator(
            close=df1.close, opn=df1.open, window=10, pow1=2, pow2=30).kama().iloc[-1]

    # Rate Of Change
    if f"{colprefix}momentum_roc" in df_columns:
        features_dict[f"{colprefix}momentum_roc"] = ROCIndicator(
            close=df1.close, window=12).roc().iloc[-1]

    # Percentage Price Oscillator
    ppo_list = [f"{colprefix}momentum_ppo", f"{colprefix}momentum_ppo_signal", f"{colprefix}momentum_ppo_hist"]
    flag = []
    for item in ppo_list:
        flag.append(True) if item in df_columns else flag.append(False)

    if True in flag:
        indicator_ppo = PercentagePriceOscillator(close=df1.close, window_slow=26, window_fast=12, window_sign=9)
        values_list = [indicator_ppo.ppo().iloc[-1], indicator_ppo.ppo_signal().iloc[-1],
                       indicator_ppo.ppo_hist().iloc[-1]]
        for i, f in enumerate(flag):
            if f is True:
                features_dict[ppo_list[i]] = values_list[i]


    # Percentage Volume Oscillator
    pvo_list = [f"{colprefix}momentum_pvo", f"{colprefix}momentum_pvo_signal", f"{colprefix}momentum_pvo_hist"]
    flag = []
    for item in pvo_list:
        flag.append(True) if item in df_columns else flag.append(False)

    if True in flag:
        indicator_pvo = PercentageVolumeOscillator(volume=log_vol1, window_slow=26, window_fast=12, window_sign=9)
        values_list = [indicator_pvo.pvo().iloc[-1], indicator_pvo.pvo_signal().iloc[-1],
                       indicator_pvo.pvo_hist().iloc[-1]]
        for i, f in enumerate(flag):
            if f is True:
                features_dict[pvo_list[i]] = values_list[i]
    return features_dict


def add_all_stationary_ta_features(df, opn: str, high: str, low: str, close: str, volume='None', up_first='None',
                                   fillna=False, colprefix: str = ""):
    """Add all technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        opn (str): Name of 'open' column.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        up_first (str|None): Name of 'up_first' column
        fillna (bool): if True, fill nan values.
        colprefix (str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    if volume != 'None':
        df = add_volume_ta(df=df, opn=opn, high=high, low=low, close=close, volume=volume, fillna=fillna,
                           colprefix=colprefix)
    df = add_volatility_ta(df=df, opn=opn, high=high, low=low, close=close, up_first=up_first, fillna=fillna,
                           colprefix=colprefix)
    df = add_trend_ta(df=df, opn=opn, high=high, low=low, close=close, fillna=fillna, colprefix=colprefix)
    df = add_momentum_ta(df=df, opn=opn, high=high, low=low, close=close, volume=volume, fillna=fillna,
                         colprefix=colprefix)
    return df


def online_stationary_ta_features(df, df_columns, colprefix: str = ""):
    """Add all technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        opn (float): Name of 'open' column.
        high (float): Name of 'high' column.
        low (float): Name of 'low' column.
        close (float): Name of 'close' column.
        volume (float): Name of 'volume' column.
        up_first (str|None): Name of 'up_first' column
        colprefix (str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    features_dict = {}
    features_dict = online_diffs(df=df, df_columns=df_columns, features_dict=features_dict, colprefix=colprefix)
    features_dict = online_volume_ta(df=df, df_columns=df_columns, features_dict=features_dict, colprefix=colprefix)
    features_dict = online_volatility_ta(df=df, df_columns=df_columns, features_dict=features_dict, colprefix=colprefix)
    features_dict = online_trend_ta(df=df, df_columns=df_columns, features_dict=features_dict, colprefix=colprefix)
    features_dict = online_momentum_ta(df=df, df_columns=df_columns, features_dict=features_dict, colprefix=colprefix)
    return features_dict
