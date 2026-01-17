import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtest import Backtester


def generate_synthetic_data(days=30, volatility=0.015):
    """Генерация реалистичных МИНУТНЫХ данных акций"""

    # Генерируем временной ряд (МИНУТНЫЕ свечи)
    minutes_per_day = 520  # 8ч 40мин торговли
    total_minutes = days * minutes_per_day

    dates = pd.date_range(
        end=datetime.now(),
        periods=total_minutes,
        freq='min'
    )

    # Начальная цена
    price = 280.0
    prices = [price]

    # Генерируем цены
    np.random.seed(42)

    # Масштабируем волатильность для минутных свечей
    minute_volatility = volatility / np.sqrt(520)

    for i in range(len(dates) - 1):
        trend = 0.00001
        change = np.random.normal(trend, minute_volatility)
        price = price * (1 + change)
        prices.append(max(price, 100))

    # Создаем OHLCV
    df = pd.DataFrame({
        'time': dates,
        'close': prices
    })

    df['open'] = df['close'].shift(1).fillna(df['close'])

    random_high = np.random.uniform(0.0001, 0.0005, len(df))
    random_low = np.random.uniform(0.0001, 0.0005, len(df))

    df['high'] = df[['open', 'close']].max(axis=1) * (1 + random_high)
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - random_low)
    df['volume'] = np.random.randint(1000, 5000, len(df))

    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    df.set_index('time', inplace=True)

    return df


def main():
    """Запустить демо бэктест на МИНУТНЫХ данных"""
    print("=" * 80)
    print("DEMO BACKTEST - ST Sentiment Trading Strategy")
    print("MINUTE-BY-MINUTE TRADING (Scalping)")
    print("=" * 80)
    print("\n⚡ Testing on 1-MINUTE candles for high-frequency trading\n")

    # Генерируем минутные данные за 30 дней
    print("Generating 1-minute candles (30 trading days)...")
    df = generate_synthetic_data(days=30, volatility=0.015)

    print(f"✅ Generated {len(df):,} minute candles")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f} RUB")
    print(f"Initial price: {df['close'].iloc[0]:.2f} RUB")
    print(f"Final price: {df['close'].iloc[-1]:.2f} RUB")
    print(f"Buy & Hold return: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")
    print(f"\nTotal trading time: ~{len(df) / 520:.1f} days ({len(df):,} minutes)")

    # Запускаем бэктест
    print("\n" + "=" * 80)
    print("Running backtest with minute-by-minute strategy...")
    print("=" * 80 + "\n")

    backtester = Backtester(initial_capital=100000)
    backtester.run_backtest(df, ticker='SBER (1-min)')

    print("\n" + "=" * 80)
    print("ℹ️  NOTES:")
    print("=" * 80)
    print("• This used 1-MINUTE candles for scalping simulation")
    print("• Real market data available during trading hours (10:00-18:40 MSK)")
    print("• Higher frequency = more opportunities but also more risk")
    print("• Commission costs will be higher with more trades")
    print("=" * 80)


if __name__ == "__main__":
    main()