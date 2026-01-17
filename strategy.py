import pandas as pd
import numpy as np
from typing import Optional, Literal
import config


class TrendStrategy:
    """Трендовая торговая стратегия с улучшенными фильтрами"""

    def __init__(self):
        self.name = "ST Sentiment Trading Strategy (Enhanced)"

    def check_long_conditions(self, df: pd.DataFrame, index: int = -1) -> bool:
        """Проверка условий для покупки (LONG) - УЛУЧШЕННАЯ"""
        row = df.iloc[index]

        # Проверяем наличие всех необходимых индикаторов
        required_cols = ['ema_short', 'ema_medium', 'ema_long', 'rsi', 'adx',
                         'macd', 'macd_signal', 'volume', 'volume_avg', 'close']
        if not all(col in df.columns for col in required_cols):
            return False

        # Условие 1: Восходящий тренд (EMA выстроены правильно)
        trend_up = (row['ema_short'] > row['ema_medium'] > row['ema_long'])

        # Условие 2: Сильный тренд (ADX)
        strong_trend = row['adx'] > config.ADX_THRESHOLD

        # Условие 3: RSI в нормальной зоне (не перекуплен) - УЖЕСТОЧЕНО
        rsi_ok = 40 < row['rsi'] < config.RSI_OVERBOUGHT

        # Условие 4: MACD бычье пересечение или выше сигнальной линии
        macd_bullish = row['macd'] > row['macd_signal']

        # Условие 5: Цена откатила к EMA_SHORT или чуть ниже (возможность входа)
        price_near_ema = (row['close'] >= row['ema_short'] * 0.995)  # В пределах 0.5%

        # Условие 6: Объем значительно выше среднего - УЖЕСТОЧЕНО
        volume_high = row['volume'] > row['volume_avg'] * 1.3  # Было 1.1

        # Условие 7: Цена должна быть выше EMA_LONG (сильный восходящий тренд) - НОВОЕ
        strong_uptrend = row['close'] > row['ema_long']

        # Условие 8: MACD histogram положительный - НОВОЕ
        macd_hist_positive = row['macd_hist'] > 0 if 'macd_hist' in df.columns else True

        # Все условия должны выполняться
        return all([
            trend_up,
            strong_trend,
            rsi_ok,
            macd_bullish,
            price_near_ema,
            volume_high,
            strong_uptrend,
            macd_hist_positive
        ])

    def check_short_conditions(self, df: pd.DataFrame, index: int = -1) -> bool:
        """Проверка условий для продажи (SHORT) - УЛУЧШЕННАЯ"""
        row = df.iloc[index]

        # Проверяем наличие всех необходимых индикаторов
        required_cols = ['ema_short', 'ema_medium', 'ema_long', 'rsi', 'adx',
                         'macd', 'macd_signal', 'volume', 'volume_avg', 'close']
        if not all(col in df.columns for col in required_cols):
            return False

        # Условие 1: Нисходящий тренд (EMA выстроены в обратном порядке)
        trend_down = (row['ema_short'] < row['ema_medium'] < row['ema_long'])

        # Условие 2: Сильный тренд (ADX)
        strong_trend = row['adx'] > config.ADX_THRESHOLD

        # Условие 3: RSI в нормальной зоне (не перепродан) - УЖЕСТОЧЕНО
        rsi_ok = config.RSI_OVERSOLD < row['rsi'] < 60

        # Условие 4: MACD медвежье пересечение или ниже сигнальной линии
        macd_bearish = row['macd'] < row['macd_signal']

        # Условие 5: Цена откатила к EMA_SHORT или чуть выше
        price_near_ema = (row['close'] <= row['ema_short'] * 1.005)  # В пределах 0.5%

        # Условие 6: Объем значительно выше среднего - УЖЕСТОЧЕНО
        volume_high = row['volume'] > row['volume_avg'] * 1.3  # Было 1.1

        # Условие 7: Цена должна быть ниже EMA_LONG (сильный нисходящий тренд) - НОВОЕ
        strong_downtrend = row['close'] < row['ema_long']

        # Условие 8: MACD histogram отрицательный - НОВОЕ
        macd_hist_negative = row['macd_hist'] < 0 if 'macd_hist' in df.columns else True

        # Все условия должны выполняться
        return all([
            trend_down,
            strong_trend,
            rsi_ok,
            macd_bearish,
            price_near_ema,
            volume_high,
            strong_downtrend,
            macd_hist_negative
        ])

    def calculate_stop_loss(self, entry_price: float, direction: Literal['long', 'short'],
                           atr: Optional[float] = None) -> float:
        """Расчет уровня стоп-лосса - УЛУЧШЕННЫЙ"""
        if direction == 'long':
            # Стоп-лосс ниже цены входа
            if atr and atr > 0:
                # Используем 2.5 ATR для более широкого стопа
                stop_loss = entry_price - (atr * 2.5)  # Было 2.0
            else:
                stop_loss = entry_price * (1 - config.STOP_LOSS_PERCENT)
        else:  # short
            # Стоп-лосс выше цены входа
            if atr and atr > 0:
                stop_loss = entry_price + (atr * 2.5)  # Было 2.0
            else:
                stop_loss = entry_price * (1 + config.STOP_LOSS_PERCENT)

        return stop_loss

    def calculate_take_profit(self, entry_price: float, stop_loss: float,
                             direction: Literal['long', 'short']) -> float:
        """Расчет уровня тейк-профита (Risk/Reward ratio) - УЛУЧШЕННЫЙ"""
        risk = abs(entry_price - stop_loss)
        reward = risk * config.TAKE_PROFIT_RATIO  # Теперь 2.5 вместо 2.0

        if direction == 'long':
            take_profit = entry_price + reward
        else:  # short
            take_profit = entry_price - reward

        return take_profit

    def calculate_position_size(self, capital: float, entry_price: float,
                               stop_loss: float) -> int:
        """Расчет размера позиции на основе риск-менеджмента - УЛУЧШЕННЫЙ"""
        # Риск на сделку в рублях
        risk_amount = capital * config.RISK_PER_TRADE

        # Риск на 1 контракт
        risk_per_contract = abs(entry_price - stop_loss)

        # Количество контрактов
        if risk_per_contract > 0:
            position_size = int(risk_amount / risk_per_contract)
        else:
            position_size = 1

        # Ограничиваем минимум 1 контракт, максимум не более 10% капитала
        max_position = int(capital * 0.1 / entry_price) if entry_price > 0 else 1
        position_size = max(1, min(position_size, max_position))

        return position_size

    def should_exit_position(self, entry_price: float, current_price: float,
                            stop_loss: float, take_profit: float,
                            direction: Literal['long', 'short'],
                            trailing_stop_price: Optional[float] = None) -> tuple[bool, str]:
        """Проверка условий выхода из позиции - УЛУЧШЕННАЯ с трейлинг-стопом"""

        if direction == 'long':
            # Проверяем стоп-лосс
            if current_price <= stop_loss:
                return True, "Stop Loss"

            # Проверяем тейк-профит
            if current_price >= take_profit:
                return True, "Take Profit"

            # Проверяем трейлинг-стоп (если включен)
            if config.TRAILING_STOP and trailing_stop_price:
                if current_price <= trailing_stop_price:
                    return True, "Trailing Stop"

        else:  # short
            # Проверяем стоп-лосс
            if current_price >= stop_loss:
                return True, "Stop Loss"

            # Проверяем тейк-профит
            if current_price <= take_profit:
                return True, "Take Profit"

            # Проверяем трейлинг-стоп (если включен)
            if config.TRAILING_STOP and trailing_stop_price:
                if current_price >= trailing_stop_price:
                    return True, "Trailing Stop"

        return False, ""

    def update_trailing_stop(self, entry_price: float, current_price: float,
                            current_trailing_stop: Optional[float],
                            direction: Literal['long', 'short']) -> float:
        """Обновление трейлинг-стопа - НОВОЕ"""

        if not config.TRAILING_STOP:
            return None

        if direction == 'long':
            # Для лонга трейлинг-стоп движется вверх
            new_trailing_stop = current_price * (1 - config.TRAILING_STOP_PERCENT)

            # Трейлинг-стоп только повышается, никогда не понижается
            if current_trailing_stop is None:
                return new_trailing_stop
            else:
                return max(current_trailing_stop, new_trailing_stop)

        else:  # short
            # Для шорта трейлинг-стоп движется вниз
            new_trailing_stop = current_price * (1 + config.TRAILING_STOP_PERCENT)

            # Трейлинг-стоп только понижается, никогда не повышается
            if current_trailing_stop is None:
                return new_trailing_stop
            else:
                return min(current_trailing_stop, new_trailing_stop)

    def get_signal(self, df: pd.DataFrame) -> dict:
        """Получить торговый сигнал - УЛУЧШЕННЫЙ"""
        signal = {
            'action': None,  # 'buy', 'sell', или None
            'price': df.iloc[-1]['close'],
            'stop_loss': None,
            'take_profit': None,
            'position_size': None,
            'reason': '',
            'confidence': 0.0  # НОВОЕ: уровень уверенности в сигнале
        }

        # Проверяем условия для Long
        if self.check_long_conditions(df):
            atr = df.iloc[-1].get('atr', None)
            stop_loss = self.calculate_stop_loss(signal['price'], 'long', atr)
            take_profit = self.calculate_take_profit(signal['price'], stop_loss, 'long')

            signal['action'] = 'buy'
            signal['stop_loss'] = stop_loss
            signal['take_profit'] = take_profit
            signal['reason'] = 'Strong uptrend: EMA aligned + ADX strong + MACD bullish + high volume'
            signal['confidence'] = self._calculate_confidence(df, 'long')

        # Проверяем условия для Short
        elif self.check_short_conditions(df):
            atr = df.iloc[-1].get('atr', None)
            stop_loss = self.calculate_stop_loss(signal['price'], 'short', atr)
            take_profit = self.calculate_take_profit(signal['price'], stop_loss, 'short')

            signal['action'] = 'sell'
            signal['stop_loss'] = stop_loss
            signal['take_profit'] = take_profit
            signal['reason'] = 'Strong downtrend: EMA aligned + ADX strong + MACD bearish + high volume'
            signal['confidence'] = self._calculate_confidence(df, 'short')

        return signal

    def _calculate_confidence(self, df: pd.DataFrame, direction: str) -> float:
        """Расчет уровня уверенности в сигнале (0.0 - 1.0) - НОВОЕ"""
        row = df.iloc[-1]
        confidence = 0.0

        if direction == 'long':
            # ADX сила (чем выше, тем лучше)
            if row['adx'] > 40:
                confidence += 0.3
            elif row['adx'] > 30:
                confidence += 0.2
            elif row['adx'] > 25:
                confidence += 0.1

            # RSI в оптимальной зоне
            if 45 < row['rsi'] < 60:
                confidence += 0.3
            elif 40 < row['rsi'] < 70:
                confidence += 0.2

            # Объем
            volume_ratio = row['volume'] / row['volume_avg']
            if volume_ratio > 1.5:
                confidence += 0.3
            elif volume_ratio > 1.3:
                confidence += 0.2

            # MACD histogram растет
            if 'macd_hist' in df.columns and len(df) > 1:
                if row['macd_hist'] > df.iloc[-2]['macd_hist']:
                    confidence += 0.1

        else:  # short
            # Аналогично для short
            if row['adx'] > 40:
                confidence += 0.3
            elif row['adx'] > 30:
                confidence += 0.2
            elif row['adx'] > 25:
                confidence += 0.1

            if 40 < row['rsi'] < 55:
                confidence += 0.3
            elif 30 < row['rsi'] < 60:
                confidence += 0.2

            volume_ratio = row['volume'] / row['volume_avg']
            if volume_ratio > 1.5:
                confidence += 0.3
            elif volume_ratio > 1.3:
                confidence += 0.2

            if 'macd_hist' in df.columns and len(df) > 1:
                if row['macd_hist'] < df.iloc[-2]['macd_hist']:
                    confidence += 0.1

        return min(confidence, 1.0)  # Ограничиваем максимум 1.0