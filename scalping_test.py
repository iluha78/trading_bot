import pandas as pd
import numpy as np
from datetime import datetime

from backtest import Backtester
from demo_backtest import generate_synthetic_data
import config


def test_scalping(ticker, days=30):
    """Тест скальпинговой стратегии на минутных данных"""

    try:
        # Генерируем минутные данные
        df = generate_synthetic_data(days=days, volatility=0.015)

        # Бэктест без вывода
        import sys, io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        import matplotlib
        matplotlib.use('Agg')

        backtester = Backtester(initial_capital=100000)
        backtester.run_backtest(df, ticker=ticker)

        sys.stdout = old_stdout

        # Статистика
        if backtester.trades:
            trades_df = pd.DataFrame(backtester.trades)
            return_pct = ((backtester.capital - 100000) / 100000) * 100
            winning = trades_df[trades_df['pnl'] > 0]
            win_rate = (len(winning) / len(trades_df)) * 100 if len(trades_df) > 0 else 0

            # Считаем доходность в день
            daily_return = return_pct / days

            # Экстраполяция на год (252 торговых дня)
            annual_return = daily_return * 252

            return {
                'ticker': ticker,
                'return_30d': return_pct,
                'return_annual': annual_return,  # ГОДОВАЯ ДОХОДНОСТЬ
                'trades': len(trades_df),
                'trades_per_day': len(trades_df) / days,
                'win_rate': win_rate,
                'final_capital': backtester.capital,
            }
        else:
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    """Тест скальпинга на всех акциях"""

    print("=" * 80)
    print("SCALPING STRATEGY TEST (1-Minute Candles)")
    print("=" * 80)
    print(f"\nTesting {len(config.INSTRUMENTS)} instruments on 30-day period...")
    print("Extrapolating to annual returns (252 trading days)\n")

    results = []

    for idx, ticker in enumerate(config.INSTRUMENTS.keys(), 1):
        print(f"[{idx}/{len(config.INSTRUMENTS)}] Testing {ticker}...", end=' ')
        result = test_scalping(ticker, days=30)

        if result:
            results.append(result)
            print(f"30d: {result['return_30d']:+.2f}% | Annual: {result['return_annual']:+.1f}% | "
                  f"Trades: {result['trades']} ({result['trades_per_day']:.1f}/day)")
        else:
            print("No trades")

    if not results:
        print("\nNo results!")
        return

    # Анализ
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('return_annual', ascending=False)

    print("\n" + "=" * 80)
    print("SCALPING RESULTS (Sorted by Annual Return):")
    print("=" * 80)
    print(f"\n{'Ticker':<8} {'30d Return':<12} {'Annual Return':<15} {'Trades':<8} {'Trades/Day':<12} {'Win Rate':<10}")
    print("-" * 80)

    for _, row in results_df.iterrows():
        print(f"{row['ticker']:<8} {row['return_30d']:>+10.2f}% {row['return_annual']:>+13.1f}% "
              f"{int(row['trades']):>6} {row['trades_per_day']:>10.1f} {row['win_rate']:>8.1f}%")

    # Статистика
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Average 30-day return: {results_df['return_30d'].mean():+.2f}%")
    print(f"Average ANNUAL return: {results_df['return_annual'].mean():+.1f}%")
    print(f"Best annual return: {results_df['return_annual'].max():+.1f}% ({results_df.loc[results_df['return_annual'].idxmax(), 'ticker']})")
    print(f"Average trades per day: {results_df['trades_per_day'].mean():.1f}")
    print(f"Total trades (30 days): {results_df['trades'].sum()}")

    # Рекомендации
    top_3 = results_df.head(3)
    print("\n" + "=" * 80)
    print("🏆 TOP-3 FOR SCALPING:")
    print("=" * 80)
    for _, row in top_3.iterrows():
        print(f"{row['ticker']}: {row['return_annual']:+.1f}% annual "
              f"({row['trades_per_day']:.1f} trades/day)")

    print("\n⚠️  Note: These are PROJECTED annual returns based on 30-day performance")
    print("Real results may vary. Test in sandbox before live trading!")

    results_df.to_csv('scalping_results.csv', index=False)
    print("\n📊 Results saved to 'scalping_results.csv'")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')

    main()