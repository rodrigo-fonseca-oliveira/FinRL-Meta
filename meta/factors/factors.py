import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm


def filter_Nan(df):
    """
    Count of NaNs for each feature
    """
    naCount_dict = {}
    for col in df.columns.values:
        if df[col].dtypes == float:
            naCount_dict[col] = len(np.where(np.isnan(df[col]).values)[0])
    for i in naCount_dict:
        if naCount_dict[i] > df.shape[0] / 10:
            print(i, naCount_dict[i])
    return naCount_dict


def del_Nan(data, columns):
    """
    Retain specified factors and remove items with NaN from them
    """
    df = data[columns]
    df.dropna(axis=1, how="all", inplace=True)
    df.dropna(axis=0, how="any", inplace=True)
    return df


def pearson_corr(df_, target):
    """
    Calculate the Pearson correlation coefficient between the factor and the target value
    """
    Pearson_dict = {}
    df_.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df_.dropna(axis=1, how="all")
    df = df_.dropna(axis=0, how="any")
    for i in df.columns.values:
        if (
            (type(df[i].values[-1]) == float or type(df[i].values[-1]) == np.float64)
            and i != "alpha084"
            and i != "alpha191-017"
        ):
            Pearson_dict[i] = scipy.stats.pearsonr(df[target].values, df[i].values)[0]

    df_Pearson = pd.DataFrame(data=Pearson_dict, index=[0]).T
    return abs(df_Pearson).sort_values(by=[0], ascending=False)


def spearmanr_corr(df_, target):
    """
    Calculate the Spearman correlation coefficient between the factor and the target value
    """
    Spearmanr_dict = {}
    df_.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df_.dropna(axis=1, how="all")
    df = df_.dropna(axis=0, how="any")
    for i in df.columns.values:
        if (
            (type(df[i].values[-1]) == float or type(df[i].values[-1]) == np.float64)
            and i != "alpha084"
            and i != "alpha191-017"
        ):
            Spearmanr_dict[i] = scipy.stats.spearmanr(df[target].values, df[i].values)[0]

    df_Spearmanr = pd.DataFrame(data=Spearmanr_dict, index=[0]).T
    return abs(df_Spearmanr).sort_values(by=[0], ascending=False)


def series_sum(S, N):  
    """
    Calculate the cumulative sum of the sequence for N days, return the sequence. If N=0, sum all in sequence.
    """
    return (
        pd.Series(S).rolling(N).sum().values if N > 0 else pd.Series(S).cumsum().values
    )


def ref(S, N=1):  
    """
    Shift the entire sequence down by N, return the sequence (shifting will produce NAN).
    """
    return pd.Series(S).shift(N).values


def ma(S, N):  
    """
    Calculate the N-day simple moving average of the sequence, return the sequence.
    """
    return pd.Series(S).rolling(N).mean().values


def ema(S, N):  
    """
    Exponential moving average. For accuracy, S should be > 4*N. EMA requires at least 120 periods.
    """
    return pd.Series(S).ewm(span=N, adjust=False).mean().values


def avedev(S, N):  
    """
    Average absolute deviation (the average value of the absolute difference between the sequence and its average).
    """
    return pd.Series(S).rolling(N).apply(lambda x: (np.abs(x - x.mean())).mean()).values


def std(S, N):  
    """
    Calculate the N-day standard deviation of the sequence, return the sequence.
    """
    return pd.Series(S).rolling(N).std(ddof=0).values


def llv(S, N):  
    """
    Lowest closing price of the last N days.
    """
    return pd.Series(S).rolling(N).min().values


def hhv(S, N):  
    """
    Highest closing price of the last N days.
    """
    return pd.Series(S).rolling(N).max().values


def sma(S, N, M=1):  
    """
    Chinese-style SMA. It requires at least 120 periods for accuracy (Snowball requires 180 periods).
    """
    return pd.Series(S).ewm(alpha=M / N, adjust=False).mean().values  # com=N-M/M


def atr(CLOSE, HIGH, LOW, N=20):  
    """
    Average True Range for N days.
    """
    TR = np.maximum(
        np.maximum((HIGH - LOW), np.abs(ref(CLOSE, 1) - HIGH)),
        np.abs(ref(CLOSE, 1) - LOW),
    )
    return ma(TR, N)



def dma(S, A):  # 求S的动态移动平均，A作平滑因子,必须 0<A<1  (此为核心函数，非指标）
    if isinstance(A, (int, float)):
        return pd.Series(S).ewm(alpha=A, adjust=False).mean().values
    A = np.array(A)
    A[np.isnan(A)] = 1.0
    Y = np.zeros(len(S))
    Y[0] = S[0]
    for i in range(1, len(S)):
        Y[i] = A[i] * S[i] + (1 - A[i]) * Y[i - 1]  # A支持序列 by jqz1226
    return Y


class MomentumFactors:
    """
    Momentum-related factors
    """

    # 5-day Deviation Rate 'ic_mean': '-0.045657'
    def bias_5_days(close, N=5):
        # (Close price - N-day simple average of close price) / N-day simple average of close price * 100, here N=5
        mac = ma(close, N)
        return (close - mac) / (mac * 100)

    # 10-day Deviation Rate 'ic_mean': '-0.043967'
    def bias_10_days(close, N=10):
        # (Close price - N-day simple average of close price) / N-day simple average of close price * 100, here N=10
        mac = ma(close, N)
        return (close - mac) / (mac * 100)

    # 60-day Deviation Rate 'ic_mean': '-0.039533'
    def bias_60_days(close, N=60):
        # (Close price - N-day simple average of close price) / N-day simple average of close price * 100, here N=60
        mac = ma(close, N)
        return (close - mac) / (mac * 100)

    # Current stock price divided by the average stock price of the past month minus 1 'ic_mean': '-0.039303'
    def price_1_month(close, N=21):
        # Current close price / mean of the past one month's (21 days) close price -1
        return close / close.rolling(N).mean() - 1

    # Current stock price divided by the average stock price of the past three months minus 1 'ic_mean': '-0.034927'
    def price_3_monthes(close, N=61):
        # Current close price / mean of the past three months' (61 days) close price -1
        return close / close.rolling(N).mean() - 1

    # 6-day Price Rate of Change 'ic_mean': '-0.030587'
    def roc_6_days(close, N=6):
        # AX = Today's close price - Close price 6 days ago
        # BX = Close price 6 days ago
        # ROC = AX/BX*100
        BX = close.shift(N)
        AX = close - BX
        return AX / (BX * 100)

    # 12-day Price Rate of Change 'ic_mean': '-0.034748'
    def roc_12_days(close, N=12):
        # AX = Today's close price - Close price 12 days ago
        # BX = Close price 12 days ago
        # ROC = AX/BX*100
        BX = close.shift(N)
        AX = close - BX
        return AX / (BX * 100)

    # 20-day Price Rate of Change 'ic_mean': '-0.031276'
    def roc_20_days(close, N=20):
        # AX = Today's close price - Close price 20 days ago
        # BX = Close price 20 days ago
        # ROC = AX/BX*100
        BX = close.shift(N)
        AX = close - BX
        return AX / (BX * 100)

    # Single Day Price and Volume Trend 'ic_mean': '-0.051037'
    def single_day_vpt(df):
        # (Today's close price - Yesterday's close price) / Yesterday's close price * Today's trading volume (adjusted for pre-close)
        sft = df["close_price"].shift(1)
        return (df["close_price"] - sft) / sft * df["volume"]

    # Single Day Price and Volume Trend 6-day Average 'ic_mean': '-0.032458'
    def single_day_vpt_6(df):
        # 6-day moving average of single_day_VPT
        sft = df["close_price"].shift(1)
        return pd.Series(ma((df["close_price"] - sft) / sft * df["volume"], 6))

    # Single Day Price and Volume Trend 12-day Average 'ic_mean': '-0.031016'
    def single_day_vpt_12(df):
        # 12-day moving average of single_day_VPT
        sft = df["close_price"].shift(1)
        return pd.Series(ma((df["close_price"] - sft) / sft * df["volume"], 12))

    # 10-day Commodity Channel Index 'ic_mean': '-0.038179'
    def cci_10_days(df, N=10):
        # CCI = (TYP - N-day moving average of TYP) / (0.015 * Average absolute deviation of TYP over N days)
        # TYP = (HIGH + LOW + CLOSE) / 3
        TYP = (df["high_price"] + df["low_price"] + df["close_price"]) / 3
        return (TYP - ma(TYP, N)) / (0.015 * avedev(TYP, N))

    # 15-day Commodity Channel Index 'ic_mean': '-0.035973'
    def cci_15_days(df, N=15):
        # Similar explanation as above, just with N=15
        TYP = (df["high_price"] + df["low_price"] + df["close_price"]) / 3
        return (TYP - ma(TYP, N)) / (0.015 * avedev(TYP, N))

    # 20-day Commodity Channel Index 'ic_mean': '-0.033437'
    def cci_20_days(df, N=20):
        # Similar explanation as above, just with N=20
        TYP = (df["high_price"] + df["low_price"] + df["close_price"]) / 3
        return (TYP - ma(TYP, N)) / (0.015 * avedev(TYP, N))

    # Current trading volume compared to the average trading volume of the past month times the average return of the past 20 days 'ic_mean': '-0.032789'
    def volume_1_month(df, N=21):
        # Today's trading volume / mean of the past 20 days trading volume * mean of the past 20 days return
        return (
            df["volume"]
            / df["volume"].rolling(N).mean()
            * df["target"].rolling(N).mean()
        )

    # Bullish Power 'ic_mean': '-0.039968'
    def bull_power(df, timeperiod=13):
        return (df["high_price"] - ema(df["close_price"], timeperiod)) / df[
            "close_price"
        ]


class EmotionFactors:
    """
    Emotion Factors
    """

    # Turnover Rate: Trading volume over a period / Total number of issued shares × 100%
    # 5-day average turnover rate 'ic_mean': '-0.044'
    def vol_5_days(S, total_volume, N=5):
        # 5-day average turnover rate
        S = S / total_volume
        return pd.Series(S).rolling(N).mean()

    # 10-day average turnover rate 'ic_mean': '-0.040'
    def vol_10_days(S, total_volume, N=10):
        # 10-day average turnover rate
        S = S / total_volume
        return pd.Series(S).rolling(N).mean()

    # 20-day average turnover rate 'ic_mean': '-0.035'
    def vol_20_days(S, total_volume, N=20):
        # 20-day average turnover rate in percentage
        S = S / total_volume
        return pd.Series(S).rolling(N).mean()

    # Ratio of 5-day average turnover to 120-day average turnover 'ic_mean': '-0.039'
    def davol_5_days(S):
        # 5-day average turnover / 120-day average turnover
        return EmotionFactors.vol_5_days(S) / EmotionFactors.vol_5_days(S, N=120)

    # Ratio of 10-day average turnover to 120-day average turnover 'ic_mean': '-0.033'
    def davol_10_days(S):
        # 10-day average turnover / 120-day average turnover
        return EmotionFactors.vol_10_days(S) / EmotionFactors.vol_5_days(S, N=120)

    # 10-day volume standard deviation 'ic_mean': '-0.037'
    def vstd_10_days(volume, N=10):
        # 10-day volume standard deviation
        return pd.Series(std(volume, N))

    # 20-day volume standard deviation 'ic_mean': '-0.033'
    def vstd_20_days(volume, N=20):
        # 20-day volume standard deviation
        return pd.Series(std(volume, N))

    # Standard deviation of the 6-day trade amount 'ic_mean': '-0.044'
    def tvstd_6_days(df, N=6):
        # 6-day trade amount standard deviation
        trades = df["close_price"] * df["volume"]
        return pd.Series(std(trades, N))

    # Standard deviation of the 20-day trade amount 'ic_mean': '-0.038'
    def tvstd_20_days(df, N=20):
        # 20-day trade amount standard deviation
        trades = df["close_price"] * df["volume"]
        return pd.Series(std(trades, N))

    # 5-day exponential moving average of the volume 'ic_mean': '-0.035'
    def vema_5_days(volume, N=5):
        return pd.Series(ema(volume, N))

    # 10-day exponential moving average of the volume 'ic_mean': '-0.032'
    def vema_10_days(volume, N=10):
        return pd.Series(ema(volume, N))

    # 12-day moving average of the volume 'ic_mean': '-0.031'
    def vema_12_days(volume, N=12):
        return pd.Series(ema(volume, N))

    # Volume Oscillation 'ic_mean': '-0.039'
    def vosc(volume):
        # Difference between 'VEMA12' and 'VEMA26' divided by 'VEMA12', then multiplied by 100 to get VOSC value
        ema12 = ema(volume, 12)
        return pd.Series((ema(volume, 26) - ema12 / (ema12 * 100)))

    # 6-day volume rate of change 'ic_mean': '-0.032'
    def vroc_6_days(volume, N=6):
        # The difference between current volume and volume N days ago, divided by the volume N days ago, then multiplied by 100 to get VROC value, n=6
        sft = volume.shift(N)
        return pd.Series((volume - sft) / (sft * 100))

    # 12-day volume rate of change 'ic_mean': '-0.040'
    def vroc_12_days(volume, N=12):
        # The difference between current volume and volume N days ago, divided by the volume N days ago, then multiplied by 100 to get VROC value, n=12
        sft = volume.shift(N)
        return pd.Series((volume - sft) / (sft * 100))

    # 6-day moving average of the trade amount 'ic_mean': '-0.038'
    def tvma_6_days(df, N=6):
        # 6-day moving average of the trade amount
        trades = df["close_price"] * df["volume"]
        return pd.Series(ma(trades, N))

    # Williams Variable Displacement Volume 'ic_mean': '-0.031'
    def wvad(df, N=6):
        # (Closing price - Opening price) / (High price - Low price) * volume, summed over the past 6 trading days
        WVA = (
            (df["close_price"] - df["open_price"])
            / (df["high_price"] - df["low_price"])
            * df["volume"]
        )
        return WVA.rolling(N).sum()

    # Relative turnover rate volatility 'ic_mean': '-0.042'
    def turnover_volatility(volume, total_volume, N=20):
        # Standard deviation of turnover rate over 20 trading days
        turnover = volume / total_volume
        return pd.Series(std(turnover, N))

    # Popularity Index 'ic_mean': '-0.031'
    def ar(df, N=26):
        # AR = sum of (High price - Opening price) over N days / sum of (Opening price - Low price) over N days * 100, N is set to 26
        ho = (df["high_price"] - df["open_price"]).rolling(N).sum()
        ol = (df["open_price"] - df["low_price"]).rolling(N).sum()
        return ho / (ol * 100)


class ExtraFactors:
    """
    Special Factors
    """

    def rsrs(df, N):
        """
        RSRS Indicator.
        This function computes the RSRS based on the relationship between the high and low prices.
        """
        ans = []  # Stores regression beta values, i.e., slope.
        ans_rightdev = []  # Stores weighted beta values adjusted by the coefficient of determination.
        
        X = sm.add_constant(df["low_price"])
        model = sm.OLS(df["high_price"], X)
        result = model.fit()
        beta = result.params
        r2 = result.rsquared
        ans.append(beta)

        # Calculate standardized RSRS
        section = ans[-N:]
        mu = np.mean(section)
        sigma = np.std(section)
        zscore = (section[-1] - mu) / sigma

        # Calculate right-skewed RSRS z-score.
        return pd.Series(zscore * beta * r2)

    def vix():
        """
        VIX function.
        (Currently not implemented)
        """
        pass


class GeneralFactors:
    """
    Common Factors
    """

    @staticmethod
    def macd(CLOSE, SHORT=12, LONG=26, M=9):
        """
        MACD Indicator.
        """
        DIF = ema(CLOSE, SHORT) - ema(CLOSE, LONG)
        DEA = ema(DIF, M)
        MACD = (DIF - DEA) * 2
        return np.round(MACD, 3)

    @staticmethod
    def kdj(df, KDJ_type, N=9, M1=3, M2=3):
        """
        KDJ Indicator.
        """
        RSV = (
            (df["close_price"] - llv(df["low_price"], N))
            / (hhv(df["high_price"], N) - llv(df["low_price"], N))
            * 100
        )
        K = ema(RSV, (M1 * 2 - 1))
        if KDJ_type == "KDJ_K":
            return K
        elif KDJ_type == "KDJ_D":
            return ema(K, (M2 * 2 - 1))
        elif KDJ_type == "KDJ_J":
            D = ema(K, (M2 * 2 - 1))
            return K * 3 - D * 2

    @staticmethod
    def rsi(CLOSE, N=24):
        """
        RSI Indicator.
        """
        DIF = CLOSE - ref(CLOSE, 1)
        return np.round(sma(np.maximum(DIF, 0), N) / sma(np.abs(DIF), N) * 100, 3)

    @staticmethod
    def wr(df, N=10):
        """
        W&R (Williams %R) Indicator.
        """
        WR = (
            (hhv(df["high_price"], N) - df["close_price"])
            / (hhv(df["high_price"], N) - llv(df["low_price"], N))
            * 100
        )
        return np.round(WR, 3)

    @staticmethod
    def roll(CLOSE, BOLL_type, N=20, P=2):
        """
        Bollinger Bands Indicator.
        """
        MID = ma(CLOSE, N)
        if BOLL_type == "BOLL_mid":
            return MID
        elif BOLL_type == "BOLL_upper":
            return MID + std(CLOSE, N) * P
        elif BOLL_type == "BOLL_lower":
            return MID - std(CLOSE, N) * P

    @staticmethod
    def psy(CLOSE, PSY_type, N=12, M=6):
        """
        PSY Indicator.
        """
        PSY = series_sum(CLOSE > ref(CLOSE, 1), N) / N * 100
        if PSY_type == "PSY":
            return PSY
        elif PSY_type == "PSYMA":
            return ma(PSY, M)

    @staticmethod
    def atr(df, N=20):
        """
        ATR (Average True Range) Indicator.
        """
        TR = np.maximum(
            np.maximum(
                (df["high_price"] - df["low_price"]),
                np.abs(ref(df["close_price"], 1) - df["high_price"]),
            ),
            np.abs(ref(df["close_price"], 1) - df["low_price"]),
        )
        return ma(TR, N)

    @staticmethod
    def bbi(CLOSE, M1=3, M2=6, M3=12, M4=20):
        """
        BBI (Bulls and Bears Index) Indicator.
        """
        return (ma(CLOSE, M1) + ma(CLOSE, M2) + ma(CLOSE, M3) + ma(CLOSE, M4)) / 4
    def dmi(df, DMI_type, M1=14, M2=6):  # Dynamic index: matches with Tonghua Shun and TDX
        TR = series_sum(
            np.maximum(
                np.maximum(
                    df["high_price"] - df["low_price"],
                    np.abs(df["high_price"] - ref(df["close_price"], 1)),
                ),
                np.abs(df["low_price"] - ref(df["close_price"], 1)),
            ),
            M1,
        )
        HD = df["high_price"] - ref(df["high_price"], 1)
        LD = ref(df["low_price"], 1) - df["low_price"]
        DMP = series_sum(np.where((HD > 0) & (HD > LD), HD, 0), M1)
        DMM = series_sum(np.where((LD > 0) & (LD > HD), LD, 0), M1)
        PDI = DMP * 100 / TR
        MDI = DMM * 100 / TR
        if DMI_type == "DMI_PDI":
            return PDI
        elif DMI_type == "DMI_MDI":
            return MDI
        elif DMI_type == "DMI_ADX":
            return ma(np.abs(MDI - PDI) / (PDI + MDI) * 100, M2)
        elif DMI_type == "DMI_ADXR":
            ADX = ma(np.abs(MDI - PDI) / (PDI + MDI) * 100, M2)
            return (ADX + ref(ADX, M2)) / 2
        # return PDI, MDI, ADX, ADXR

    def taq(df, TAQ_type, N=6):  # Donchian Channel (Turtle) indicator, simplicity, bullish/bearish crossover
        UP = hhv(df["high_price"], N)
        DOWN = llv(df["low_price"], N)
        if TAQ_type == "TAQ_UP":
            return UP
        elif TAQ_type == "TAQ_DOWN":
            return DOWN
        elif TAQ_type == "TAQ_MID":
            return (UP + DOWN) / 2
        # return UP,MID,DOWN

    def ktn(df, KTN_type, N=20, M=10):  # Keltner Channel, choose N as 20 days and ATR as 10 days
        MID = ema((df["high_price"] + df["low_price"] + df["close_price"]) / 3, N)
        if KTN_type == "KTN_mid":
            return MID
        elif KTN_type == "KTN_upper":
            ATRN = atr(df["close_price"], df["high_price"], df["low_price"], M)
            return MID + 2 * ATRN
        elif KTN_type == "KTN_lower":
            ATRN = atr(df["close_price"], df["high_price"], df["low_price"], M)
            return MID - 2 * ATRN
        # return UPPER,MID,LOWER

    def trix(CLOSE, TRIX_type, M1=12, M2=20):  # Triple Exponential Moving Average
        TR = ema(ema(ema(CLOSE, M1), M1), M1)
        TRIX = (TR - ref(TR, 1)) / ref(TR, 1) * 100
        if TRIX_type == "TRIX":
            return TRIX
        elif TRIX_type == "TRMA":
            return ma(TRIX, M2)
        # return TRIX, TRMA

    def vr(df, M1=26):  # VR volume ratio
        LC = ref(df["close_price"], 1)
        return (
            series_sum(np.where(df["close_price"] > LC, df["volume"], 0), M1)
            / series_sum(np.where(df["close_price"] <= LC, df["volume"], 0), M1)
            * 100
        )

    def emv(df, EMV_type, N=14, M=9):  # Simplified Volatility Index
        VOLUME = ma(df["volume"], N) / df["volume"]
        MID = (
            100
            * (
                df["high_price"]
                + df["low_price"]
                - ref(df["high_price"] + df["low_price"], 1)
            )
            / (df["high_price"] + df["low_price"])
        )
        EMV = ma(
            MID
            * VOLUME
            * (df["high_price"] - df["low_price"])
            / ma(df["high_price"] - df["low_price"], N),
            N,
        )
        if EMV_type == "EMV":
            return EMV
        elif EMV_type == "MAEMV":
            return ma(EMV, M)
        # return EMV,MAEMV


    def dpo(CLOSE, DPO_type, M1=20, M2=10, M3=6):  # Detrended Price Oscillator
        DPO = CLOSE - ref(ma(CLOSE, M1), M2)
        if DPO_type == "DPO":
            return DPO
        elif DPO_type == "MADPO":
            return ma(DPO, M3)
        # return DPO, MADPO

    def brar(df, M1=26):  # BRAR-ARBR Sentiment Indicator
        # AR = series_sum(HIGH - OPEN, M1) / series_sum(OPEN - LOW, M1) * 100
        return (
            series_sum(np.maximum(0, df["high_price"] - ref(df["close_price"], 1)), M1)
            / series_sum(np.maximum(0, ref(df["close_price"], 1) - df["low_price"]), M1)
            * 100
        )
        # return AR, BR

    def dfma(CLOSE, N1=10, N2=50, M=10):  # Parallel Line Differential Indicator (Tongdaxin called it DMA, Tonghuashun called it new DMA)
        DIF = ma(CLOSE, N1) - ma(CLOSE, N2)
        DIFMA = ma(DIF, M)  
        return DIFMA

    def mtm(CLOSE, MTM_type, N=12, M=6):  # Momentum Indicator
        MTM = CLOSE - ref(CLOSE, N)
        if MTM_type == "MTM":
            return MTM
        elif MTM_type == "MTMMA":
            return ma(MTM, M)
        # return MTM,MTMMA

    def mass(df, MASS_type, N1=9, N2=25, M=6):  # Mass Index
        MASS = series_sum(
            ma(df["high_price"] - df["low_price"], N1)
            / ma(ma(df["high_price"] - df["low_price"], N1), N1),
            N2,
        )
        if MASS_type == "MASS":
            return MASS
        elif MASS_type == "MA_MASS":
            return ma(MASS, M)
        # return MASS,MA_MASS

    def obv(df):  # On-Balance Volume Indicator
        return (
            series_sum(
                np.where(
                    df["close_price"] > ref(df["close_price"], 1),
                    df["volume"],
                    np.where(
                        df["close_price"] < ref(df["close_price"], 1),
                        -df["volume"],
                        0,
                    ),
                ),
                0,
            )
            / 10000
        )

    def mfi(df, N=14):  # MFI is the RSI of Volume
        TYP = (df["high_price"] + df["low_price"] + df["close_price"]) / 3
        V1 = series_sum(
            np.where(TYP > ref(TYP, 1), TYP * df["volume"], 0), N
        ) / series_sum(np.where(TYP < ref(TYP, 1), TYP * df["volume"], 0), N)
        return 100 - (100 / (1 + V1))

    def asi(df, ASI_type, M1=26, M2=10):  # Swing Index
        LC = ref(df["close_price"], 1)
        AA = np.abs(df["high_price"] - LC)
        BB = np.abs(df["low_price"] - LC)
        CC = np.abs(df["high_price"] - ref(df["low_price"], 1))
        DD = np.abs(LC - ref(df["open_price"], 1))
        R = np.where(
            (AA > BB) & (AA > CC),
            AA + BB / 2 + DD / 4,
            np.where((BB > CC) & (BB > AA), BB + AA / 2 + DD / 4, CC + DD / 4),
        )
        X = (
            df["close_price"]
            - LC
            + (df["close_price"] - df["open_price"]) / 2
            + LC
            - ref(df["open_price"], 1)
        )
        SI = 16 * X / R * np.maximum(AA, BB)
        ASI = series_sum(SI, M1)
        if ASI_type == "ASI":
            return ASI
        elif ASI_type == "ASIT":
            return ma(ASI, M2)
        # return ASI,ASIT

    def xsii(df, XSII_type, N=102, M=7):  # Xue's Channel II
        AA = ma(
            (2 * df["close_price"] + df["high_price"] + df["low_price"]) / 4, 5
        )  # Only the latest version of DMA supports this as of 2021-12-4
        # TD1 = AA*N/100   TD2 = AA*(200-N) / 100
        if XSII_type == "XSII_TD1":
            return AA * N / 100
        elif XSII_type == "XSII_TD2":
            return AA * (200 - N) / 100
        elif XSII_type == "XSII_TD3":
            CC = np.abs(
                (2 * df["close_price"] + df["high_price"] + df["low_price"]) / 4
                - ma(df["close_price"], 20)
            ) / ma(df["close_price"], 20)
            BB = df["close_price"].reset_index()["close_price"]
            DD = dma(BB, CC)
            return (1 + M / 100) * DD
        elif XSII_type == "XSII_TD4":
            CC = np.abs(
                (2 * df["close_price"] + df["high_price"] + df["low_price"]) / 4
                - ma(df["close_price"], 20)
            ) / ma(df["close_price"], 20)
            BB = df["close_price"].reset_index()["close_price"]
            DD = dma(BB, CC)
            return (1 - M / 100) * DD

