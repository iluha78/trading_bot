import pandas as pd
import numpy as np
from typing import Optional, Literal
import config
from strategy import TrendStrategy


class AdvancedTrendStrategy(TrendStrategy):
    """Продвинутая стратегия с пирамидингом"""

    def __init__(self):
        super().__init__()
        self.name = "ST Sentiment Trading Strategy (Advanced + Pyramiding)"

    def check_pyramid_conditions(self, df: pd.DataFrame, entry_price: float,
                                 current_price: float, direction: str) -> bool:
        """Проверка условий для добавления к позиции (пирамидинг)"""

        if direction == 'buy':
            # Добавляем к лонгу если:
            # 1. Цена выросла на 3% от входа
            price_gain = (current_price - entry_price) / entry_price
            if price_gain < 0.03:
                return False

            # 2. Тренд все еще сильный
            if not self.check_long_conditions(df):
                return False

            # 3. RSI не перекуплен
            if df.iloc[-1]['rsi'] > 70:
                return False

            return True

        else:  # sell
            # Добавляем к шорту если:
            price_gain = (entry_price - current_price) / entry_price
            if price_gain < 0.03:
                return False

            if not self.check_short_conditions(df):
                return False

            if df.iloc[-1]['rsi'] < 30:
                return False

            return True

    def calculate_pyramid_size(self, original_size: int) -> int:
        """Размер пирамиды - половина от исходной позиции"""
        return max(1, original_size // 2)