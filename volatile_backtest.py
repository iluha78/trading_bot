import pandas as pd
import numpy as np
from datetime import datetime

from backtest import Backtester
from demo_backtest import generate_synthetic_data
import config


def backtest_volatile_instrument(ticker, days=365):
    """Бэктест волатильного инструмента"""

    # Высокая волатильность для разных секторов
    volatility_map = {
        # Технологии - ОЧЕНЬ высокая волатильность
        'YNDX': 0.030,
        'VKCO': 0.030,
        'OZON': 0.040,
        'FIXP': 0.035,

        # Нефтегаз - высокая
        'NVTK': 0.025,
        'TATN': 0.022,
        'SNGS': 0.028,

        # Металлы - высокая
        'GMKN': 0.028,
        'PLZL': 0.032,
        'PHOR': 0.025,
        'RUAL': 0.030,

        # Ритейл - средняя
        'FIVE': 0.020,
        'MGNT': 0.020,

        # Финансы
        'SBER': 0.018,
        'VTBR': 0.022,

        # Другие
        'AFKS': 0.028,
    }

    volatility = volatility_map.get(ticker, 0.025)

    try:
        # Генерируем данные с высокой волатильностью
        df = generate_synthetic_data(days=days, volatility=volatility)

        # Запускаем бэктест без вывода
        import sys, io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        import matplotlib
        matplotlib.use('Agg')

        backtester = Backtester(initial_capital=100000)
        backtester.run_backtest(df, ticker=ticker)

        sys.stdout = old_stdout

        # Собираем статистику
        if backtester.trades:
            trades_df = pd.DataFrame(backtester.trades)
            total_return = ((backtester.capital - backtester.initial_capital) /
                          backtester.initial_capital) * 100
            winning = trades_df[trades_df['pnl'] > 0]
            win_rate = (len(winning) / len(trades_df)) * 100 if len(trades_df) > 0 else 0

            return {
                'ticker': ticker,
                'volatility': volatility * 100,  # В процентах
                'return_pct': total_return,
                'final_capital': backtester.capital,
                'trades': len(trades_df),
                'win_rate': win_rate,
                'avg_win': winning['pnl'].mean() if len(winning) > 0 else 0,
                'avg_loss': trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0,
                'profit_factor': abs(winning['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0,
            }
        else:
            return {
                'ticker': ticker,
                'volatility': volatility * 100,
                'return_pct': 0,
                'final_capital': 100000,
                'trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
            }

    except Exception as e:
        print(f"Error testing {ticker}: {e}")
        return None


def main():
    """Тестирование волатильных инструментов"""

    print("=" * 80)
    print("VOLATILE INSTRUMENTS BACKTEST")
    print("Testing high-volatility stocks for maximum profit potential")
    print("=" * 80)

    volatile_instruments = {
        'YNDX': 'YNDX',
        'VKCO': 'VKCO',
        'OZON': 'OZON',
        'FIXP': 'FIXP',
        'NVTK': 'NVTK',
        'TATN': 'TATN',
        'SNGS': 'SNGS',
        'GMKN': 'GMKN',
        'PLZL': 'PLZL',
        'PHOR': 'PHOR',
        'FIVE': 'FIVE',
        'MGNT': 'MGNT',
        'SBER': 'SBER',
        'VTBR': 'VTBR',
        'AFKS': 'AFKS',
        'RUAL': 'RUAL',
    }

    print(f"\nTesting {len(volatile_instruments)} volatile instruments...")
    print("Higher volatility = more profit potential (and risk)\n")

    results = []

    for idx, ticker in enumerate(volatile_instruments.keys(), 1):
        print(f"[{idx}/{len(volatile_instruments)}] Testing {ticker}...", end=' ')
        result = backtest_volatile_instrument(ticker)

        if result:
            results.append(result)
            print(f"Volatility: {result['volatility']:.1f}% | Return: {result['return_pct']:+.2f}% | Trades: {result['trades']}")
        else:
            print("FAILED")

    if not results:
        print("\nNo results to analyze!")
        return None

    # Анализ результатов
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('return_pct', ascending=False)

    print("\n" + "=" * 80)
    print("RESULTS BY PROFITABILITY:")
    print("=" * 80)

    print("\n🚀 TOP 10 PERFORMERS:")
    print("-" * 80)
    print(f"{'Ticker':<8} {'Vol%':<6} {'Return%':<10} {'Trades':<8} {'WinRate%':<10} {'P.Factor':<10} {'Capital':<12}")
    print("-" * 80)
    for idx, row in results_df.head(10).iterrows():
        print(f"{row['ticker']:<8} {row['volatility']:<6.1f} {row['return_pct']:+9.2f} {int(row['trades']):<8} "
              f"{row['win_rate']:<9.1f} {row['profit_factor']:<9.2f} {row['final_capital']:>11,.0f}")

    print("\n📉 BOTTOM 5 PERFORMERS:")
    print("-" * 80)
    for idx, row in results_df.tail(5).iterrows():
        print(f"{row['ticker']:<8} {row['volatility']:<6.1f} {row['return_pct']:+9.2f} {int(row['trades']):<8} "
              f"{row['win_rate']:<9.1f} {row['profit_factor']:<9.2f} {row['final_capital']:>11,.0f}")

    # Статистика по волатильности
    print("\n" + "=" * 80)
    print("VOLATILITY ANALYSIS:")
    print("=" * 80)

    high_vol = results_df[results_df['volatility'] > 2.5]
    mid_vol = results_df[(results_df['volatility'] >= 2.0) & (results_df['volatility'] <= 2.5)]
    low_vol = results_df[results_df['volatility'] < 2.0]

    print(f"\nHigh Volatility (>2.5%): {len(high_vol)} instruments")
    if len(high_vol) > 0:
        print(f"  Average Return: {high_vol['return_pct'].mean():+.2f}%")
        print(f"  Average Trades: {high_vol['trades'].mean():.0f}")
        print(f"  Top: {high_vol.iloc[0]['ticker']} ({high_vol.iloc[0]['return_pct']:+.2f}%)")

    print(f"\nMedium Volatility (2.0-2.5%): {len(mid_vol)} instruments")
    if len(mid_vol) > 0:
        print(f"  Average Return: {mid_vol['return_pct'].mean():+.2f}%")
        print(f"  Average Trades: {mid_vol['trades'].mean():.0f}")

    print(f"\nLow Volatility (<2.0%): {len(low_vol)} instruments")
    if len(low_vol) > 0:
        print(f"  Average Return: {low_vol['return_pct'].mean():+.2f}%")
        print(f"  Average Trades: {low_vol['trades'].mean():.0f}")

    # Общая статистика
    total_trades = results_df['trades'].sum()
    avg_return = results_df['return_pct'].mean()
    profitable_count = len(results_df[results_df['return_pct'] > 0])

    print("\n" + "=" * 80)
    print("OVERALL STATISTICS:")
    print("=" * 80)
    print(f"Total Instruments: {len(results_df)}")
    print(f"Profitable: {profitable_count} ({profitable_count/len(results_df)*100:.1f}%)")
    print(f"Average Return: {avg_return:+.2f}%")
    print(f"Best Return: {results_df['return_pct'].max():+.2f}% ({results_df.loc[results_df['return_pct'].idxmax(), 'ticker']})")
    print(f"Worst Return: {results_df['return_pct'].min():+.2f}% ({results_df.loc[results_df['return_pct'].idxmin(), 'ticker']})")
    print(f"Total Trades: {int(total_trades)}")
    print(f"Avg Trades per Stock: {total_trades/len(results_df):.0f}")

    # Портфель
    portfolio_capital = results_df['final_capital'].sum()
    portfolio_initial = 100000 * len(results_df)
    portfolio_return = ((portfolio_capital - portfolio_initial) / portfolio_initial) * 100

    print("\n" + "=" * 80)
    print("PORTFOLIO SIMULATION:")
    print("=" * 80)
    print(f"Initial: {portfolio_initial:,.0f} RUB")
    print(f"Final: {portfolio_capital:,.0f} RUB")
    print(f"Return: {portfolio_return:+.2f}%")
    print(f"Profit: {portfolio_capital - portfolio_initial:+,.0f} RUB")

    # Рекомендации
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS:")
    print("=" * 80)

    top_10 = results_df.head(10)
    print(f"\n✅ TOP-10 for maximum profit:")
    print(f"   Average Return: {top_10['return_pct'].mean():+.2f}%")
    print(f"   Average Volatility: {top_10['volatility'].mean():.1f}%")
    print(f"   Tickers: {', '.join(top_10['ticker'].tolist())}")

    # Сохраняем
    results_df.to_csv('volatile_backtest_results.csv', index=False)
    print("\n📊 Results saved to 'volatile_backtest_results.csv'")

    # Для config.py
    print("\n" + "=" * 80)
    print("📋 RECOMMENDED FOR config.py:")
    print("=" * 80)
    print("\nINSTRUMENTS = {")
    for ticker in top_10['ticker']:
        print(f"    '{ticker}': '{ticker}',  # {results_df[results_df['ticker']==ticker].iloc[0]['return_pct']:+.2f}%")
    print("}")

    return results_df


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')

    results = main()