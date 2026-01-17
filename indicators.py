import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange


class TechnicalIndicators:
    """Класс для расчета технических индикаторов"""

    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """Экспоненциальная скользящая средняя"""
        ema = EMAIndicator(close=df[column], window=period)
        return ema.ema_indicator()

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """Индекс относительной силы"""
        rsi = RSIIndicator(close=df[column], window=period)
        return rsi.rsi()

    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                       signal: int = 9, column: str = 'close') -> tuple:
        """MACD индикатор"""
        macd = MACD(close=df[column], window_fast=fast, window_slow=slow, window_sign=signal)
        return (
            macd.macd(),           # MACD line
            macd.macd_signal(),    # Signal line
            macd.macd_diff()       # Histogram
        )

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index - сила тренда"""
        adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=period)
        return adx.adx()

    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20,
                                   std_dev: int = 2, column: str = 'close') -> tuple:
        """Bollinger Bands"""
        bb = BollingerBands(close=df[column], window=period, window_dev=std_dev)
        return (
            bb.bollinger_hband(),      # Upper band
            bb.bollinger_mavg(),       # Middle band
            bb.bollinger_lband()       # Lower band
        )

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range - волатильность"""
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=period)
        return atr.average_true_range()

    @staticmethod
    def calculate_volume_avg(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Средний объем"""
        return df['volume'].rolling(window=period).mean()

    @staticmethod
    def add_all_indicators(df: pd.DataFrame, config) -> pd.DataFrame:
        """Добавить все индикаторы к датафрейму"""
        df = df.copy()

        # EMA
        df['ema_short'] = TechnicalIndicators.calculate_ema(df, config.EMA_SHORT)
        df['ema_medium'] = TechnicalIndicators.calculate_ema(df, config.EMA_MEDIUM)
        df['ema_long'] = TechnicalIndicators.calculate_ema(df, config.EMA_LONG)

        # RSI
        df['rsi'] = TechnicalIndicators.calculate_rsi(df, config.RSI_PERIOD)

        # MACD
        macd, signal, hist = TechnicalIndicators.calculate_macd(
            df, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL
        )
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist

        # ADX
        df['adx'] = TechnicalIndicators.calculate_adx(df, config.ADX_PERIOD)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(df)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower

        # ATR
        df['atr'] = TechnicalIndicators.calculate_atr(df)

        # Volume
        df['volume_avg'] = TechnicalIndicators.calculate_volume_avg(df)

        return df