import pandas as pd
from datetime import datetime, timedelta
from tinkoff.invest import Client, CandleInterval
from tinkoff.invest.utils import now

import config
from trader import TradingBot
from backtest import Backtester


def main():
    """Запустить быстрый бэктест"""
    print("=" * 80)
    print("BACKTEST - ST Sentiment Trading Strategy")
    print("=" * 80)

    # Проверяем токены
    print(f"\nTOKEN CHECK:")
    print(f"READONLY TOKEN: {config.TINKOFF_READONLY_TOKEN[:20] if config.TINKOFF_READONLY_TOKEN else 'NOT FOUND'}...")
    print(f"REAL TOKEN: {config.TINKOFF_REAL_TOKEN[:20] if config.TINKOFF_REAL_TOKEN else 'NOT FOUND'}...")
    print(f"SANDBOX TOKEN: {config.TINKOFF_SANDBOX_TOKEN[:20] if config.TINKOFF_SANDBOX_TOKEN else 'NOT FOUND'}...")


    # Используем реальный токен для получения данных (только чтение!)
    token = config.TINKOFF_READONLY_TOKEN or config.TINKOFF_REAL_TOKEN or config.TINKOFF_SANDBOX_TOKEN
    if not token:
        print("ERROR: No token found in config!")
        return

    bot = TradingBot(token, sandbox=False)

    with Client(token) as client:
        # Тестируем на Сбербанке
        print("\nSearching for SBER...")
        instruments = client.instruments.find_instrument(query='SBER')

        sber_figi = None
        for inst in instruments.instruments:
            if inst.ticker == 'SBER':
                sber_figi = inst.figi
                print(f"Found: {inst.name} (FIGI: {inst.figi})")
                break

        if not sber_figi:
            print("SBER not found!")
            return

        # Получаем исторические данные за 3 месяца (меньше, чтобы быстрее)
        print("\nFetching historical data (365 days, hourly candles)...")
        to_time = now()
        from_time = to_time - timedelta(days=365)

        print(f"From: {from_time}")
        print(f"To: {to_time}")

        candles = []
        try:
            for candle in client.get_all_candles(
                figi=sber_figi,
                from_=from_time,
                to=to_time,
                interval=CandleInterval.CANDLE_INTERVAL_HOUR
            ):
                candles.append(candle)

                # Показываем прогресс каждые 100 свечей
                if len(candles) % 100 == 0:
                    print(f"Loaded {len(candles)} candles...")

        except Exception as e:
            print(f"Error fetching candles: {e}")

            if not candles:
                print("\nTrying with daily candles instead...")
                from_time = to_time - timedelta(days=365)

                for candle in client.get_all_candles(
                    figi=sber_figi,
                    from_=from_time,
                    to=to_time,
                    interval=CandleInterval.CANDLE_INTERVAL_DAY
                ):
                    candles.append(candle)

        if not candles:
            print("\nERROR: No candles received!")
            print("This might be because:")
            print("1. You need a token with 'Read' permissions (not sandbox)")
            print("2. Market is closed on weekends")
            print("3. SBER ticker might be delisted")
            return

        print(f"\n✅ Successfully loaded {len(candles)} candles!")

        # Конвертируем в DataFrame
        df = pd.DataFrame([{
            'time': c.time,
            'open': bot._quotation_to_float(c.open),
            'high': bot._quotation_to_float(c.high),
            'low': bot._quotation_to_float(c.low),
            'close': bot._quotation_to_float(c.close),
            'volume': c.volume
        } for c in candles])

        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")

        # Запускаем бэктест
        print("\n" + "=" * 80)
        print("Running backtest...")
        print("=" * 80 + "\n")

        backtester = Backtester(initial_capital=100000)
        backtester.run_backtest(df, ticker='SBER')


if __name__ == "__main__":
    main()