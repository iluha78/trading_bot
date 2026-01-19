import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tinkoff.invest import Client, CandleInterval
from tinkoff.invest.utils import now
import os

import config
from trader import TradingBot


def export_historical_data(ticker, figi, client, bot, days=365):
    """Выгрузка исторических данных для одной акции кроме сбер втб газпром идр"""

    print(f"Fetching data for {ticker}...", end=' ')

    try:
        # Получаем минутные свечи
        to_time = now()
        from_time = to_time - timedelta(days=days)

        candles = []
        for candle in client.get_all_candles(
            figi=figi,
            from_=from_time,
            to=to_time,
            interval=CandleInterval.CANDLE_INTERVAL_1_MIN
        ):
            candles.append(candle)

        if not candles:
            print("No data!")
            return None

        # Конвертируем в DataFrame
        df = pd.DataFrame([{
            'timestamp': candle.time,
            'open': bot._quotation_to_float(candle.open),
            'high': bot._quotation_to_float(candle.high),
            'low': bot._quotation_to_float(candle.low),
            'close': bot._quotation_to_float(candle.close),
            'volume': candle.volume,
        } for candle in candles])

        print(f"✅ {len(df):,} candles")
        return df

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def add_technical_indicators(df):
    """Добавить технические индикаторы"""
    from indicators import TechnicalIndicators

    # Добавляем все индикаторы
    df = TechnicalIndicators.add_all_indicators(df, config)

    # Добавляем дополнительные фичи для ML
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Momentum indicators
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

    # Volatility
    df['volatility_10'] = df['returns'].rolling(window=10).std()
    df['volatility_20'] = df['returns'].rolling(window=20).std()

    # Volume indicators
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    df['volume_change'] = df['volume'].pct_change()

    # Price position in range
    df['high_low_ratio'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001)

    # Trend strength
    df['ema_spread_short'] = (df['ema_short'] - df['ema_medium']) / df['ema_medium']
    df['ema_spread_long'] = (df['ema_medium'] - df['ema_long']) / df['ema_long']

    # Target variable - будущая доходность через N минут
    for horizon in [5, 10, 15, 30, 60]:
        df[f'future_return_{horizon}min'] = df['close'].shift(-horizon) / df['close'] - 1

    return df


def main():
    """Выгрузка данных для всех акций"""

    print("=" * 80)
    print("HISTORICAL DATA EXPORT TO EXCEL")
    print("=" * 80)

    # Используем readonly токен
    token = config.TINKOFF_READONLY_TOKEN or config.TINKOFF_REAL_TOKEN

    if not token:
        print("❌ No token found!")
        return

    bot = TradingBot(token, sandbox=False)

    # Создаем папку для данных
    os.makedirs('data', exist_ok=True)

    all_data = {}

    with Client(token) as client:
        print(f"\nFetching data for {len(config.INSTRUMENTS)} instruments...\n")

        for ticker in config.INSTRUMENTS.keys():
            # Получаем FIGI
            instruments = client.instruments.find_instrument(query=ticker)
            figi = None

            for inst in instruments.instruments:
                if inst.ticker == ticker:
                    figi = inst.figi
                    break

            if not figi:
                print(f"{ticker}: FIGI not found")
                continue

            # Выгружаем данные
            df = export_historical_data(ticker, figi, client, bot, days=90)

            if df is not None and len(df) > 200:
                # Добавляем индикаторы
                print(f"  Adding indicators...", end=' ')
                df = add_technical_indicators(df)
                print("✅")

                # Удаляем NaN
                df = df.dropna()

                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
                # Сохраняем
                all_data[ticker] = df

                # Экспорт в Excel
                filename = f'data/{ticker}_historical_data.xlsx'
                df.to_excel(filename, index=False)
                print(f"  Saved to {filename}")
                print()

    # Создаем общий файл со всеми акциями
    if all_data:
        print("\n" + "=" * 80)
        print("Creating combined dataset...")
        print("=" * 80)

        # Объединяем все данные
        combined_data = []
        for ticker, df in all_data.items():
            df_copy = df.copy()
            df_copy['ticker'] = ticker
            combined_data.append(df_copy)

        combined_df = pd.concat(combined_data, ignore_index=True)

        # Сохраняем в Excel
        combined_df.to_excel('data/all_stocks_combined.xlsx', index=False)
        print(f"✅ Combined data saved: {len(combined_df):,} rows")

        # Статистика
        print("\n" + "=" * 80)
        print("EXPORT SUMMARY:")
        print("=" * 80)

        for ticker, df in all_data.items():
            print(f"{ticker}: {len(df):,} rows, "
                  f"{df['timestamp'].min()} to {df['timestamp'].max()}")

        print(f"\nTotal instruments: {len(all_data)}")
        print(f"Total rows: {len(combined_df):,}")
        print(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")

        # Сохраняем также CSV для ML
        combined_df.to_csv('data/all_stocks_combined.csv', index=False)
        print(f"\n📊 Data exported to:")
        print(f"   - Individual files: data/[TICKER]_historical_data.xlsx")
        print(f"   - Combined Excel: data/all_stocks_combined.xlsx")
        print(f"   - Combined CSV: data/all_stocks_combined.csv (for ML training)")

        return all_data
    else:
        print("\n❌ No data exported!")
        return None


if __name__ == "__main__":
    data = main()