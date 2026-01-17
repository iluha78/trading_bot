import pandas as pd
import numpy as np
import joblib
import os
from typing import Optional, Literal

import config
from strategy import TrendStrategy


class PersonalizedMLStrategy:
    """Персонализированная ML-стратегия для каждой акции"""

    def __init__(self, ticker):
        self.ticker = ticker
        self.name = f"ML Strategy for {ticker}"

        # Загружаем модель для этой акции
        model_path = f'models/{ticker}_model.pkl'
        features_path = f'models/{ticker}_features.pkl'

        if os.path.exists(model_path) and os.path.exists(features_path):
            self.model = joblib.load(model_path)
            self.features = joblib.load(features_path)
            self.has_model = True
            print(f"✅ Loaded ML model for {ticker}")
        else:
            self.model = None
            self.features = None
            self.has_model = False
            print(f"⚠️  No ML model for {ticker}, using base strategy")

            # Fallback to base strategy
            self.base_strategy = TrendStrategy()

    def prepare_features(self, df: pd.DataFrame, index: int = -1) -> Optional[pd.DataFrame]:
        """Подготовка фичей для предсказания"""

        if not self.has_model:
            return None

        row = df.iloc[index]

        # Создаем DataFrame с одной строкой
        features_dict = {}
        for feature in self.features:
            if feature in df.columns:
                features_dict[feature] = [row[feature]]
            else:
                features_dict[feature] = [0]  # Default value

        return pd.DataFrame(features_dict)

    def get_ml_signal(self, df: pd.DataFrame) -> dict:
        """Получить сигнал от ML модели"""

        if not self.has_model:
            # Используем базовую стратегию
            return self.base_strategy.get_signal(df)

        # Подготавливаем фичи
        X = self.prepare_features(df)

        if X is None:
            return {'action': None, 'price': df.iloc[-1]['close'], 'confidence': 0}

        # Предсказание
        try:
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]

            signal = {
                'price': df.iloc[-1]['close'],
                'action': None,
                'confidence': 0,
                'stop_loss': None,
                'take_profit': None,
                'reason': ''
            }

            # 1 = BUY signal
            if prediction == 1:
                buy_probability = probability[1]

                # Торгуем только при высокой уверенности (>65%)
                if buy_probability > 0.65:
                    signal['action'] = 'buy'
                    signal['confidence'] = buy_probability
                    signal['reason'] = f'ML BUY signal (confidence: {buy_probability:.1%})'

                    # Рассчитываем stop loss и take profit
                    signal['stop_loss'] = signal['price'] * (1 - config.STOP_LOSS_PERCENT)
                    signal['take_profit'] = signal['price'] * (1 + config.STOP_LOSS_PERCENT * config.TAKE_PROFIT_RATIO)

            return signal

        except Exception as e:
            print(f"ML prediction error: {e}")
            return {'action': None, 'price': df.iloc[-1]['close'], 'confidence': 0}

    def get_signal(self, df: pd.DataFrame) -> dict:
        """Главный метод получения сигнала"""
        return self.get_ml_signal(df)