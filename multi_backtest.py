import pandas as pd
import numpy as np
from datetime import datetime

from backtest import Backtester
from demo_backtest import generate_synthetic_data
import config


def backtest_instrument(ticker, days=365):
    """Бэктест одного инструмента"""

    # Генерируем данные с разной волатильностью для разных секторов
    volatility_map = {
        # Высокая волатильность
        'YNDX': 0.025, 'VKCO': 0.025,

        # Средняя волатильность
        'GAZP': 0.018, 'LKOH': 0.018, 'ROSN': 0.018,
        'GMKN': 0.020, 'NLMK': 0.018, 'MAGN': 0.020,

        # Низкая волатильность
        'SBER': 0.015, 'VTBR': 0.016,
        'MTSS': 0.014, 'FEES': 0.012,
    }

    volatility = volatility_map.get(ticker, 0.015)

    try:
        # Генерируем данные
        df = generate_synthetic_data(days=days, volatility=volatility)

        # Запускаем бэктест
        backtester = Backtester(initial_capital=100000)

        # Без вывода и без графиков
        import sys, io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        # Временно отключаем графики
        import matplotlib
        matplotlib.use('Agg')  # Используем backend без GUI

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
                'return_pct': total_return,  # ИСПРАВЛЕНО: было 'return'
                'final_capital': backtester.capital,
                'trades': len(trades_df),
                'win_rate': win_rate,
                'avg_win': winning['pnl'].mean() if len(winning) > 0 else 0,
                'avg_loss': trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0,
                'max_win': trades_df['pnl'].max(),
                'max_loss': trades_df['pnl'].min(),
            }
        else:
            return {
                'ticker': ticker,
                'return_pct': 0,  # ИСПРАВЛЕНО
                'final_capital': 100000,
                'trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_win': 0,
                'max_loss': 0,
            }

    except Exception as e:
        print(f"Error testing {ticker}: {e}")
        return None


def main():
    """Тестирование всех инструментов"""

    print("=" * 80)
    print("MULTI-INSTRUMENT BACKTEST")
    print("=" * 80)
    print(f"\nTesting {len(config.INSTRUMENTS)} instruments...")
    print("This may take a few minutes...\n")

    results = []

    # ПОСЛЕДОВАТЕЛЬНОЕ тестирование (не параллельное)
    for idx, ticker in enumerate(config.INSTRUMENTS.keys(), 1):
        print(f"[{idx}/{len(config.INSTRUMENTS)}] Testing {ticker}...", end=' ')
        result = backtest_instrument(ticker)

        if result:
            results.append(result)
            print(f"Return: {result['return_pct']:+.2f}% ({result['trades']} trades)")
        else:
            print("FAILED")

    if not results:
        print("\nNo results to analyze!")
        return None

    # Анализ результатов
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('return_pct', ascending=False)  # ИСПРАВЛЕНО

    print("\n" + "=" * 80)
    print("RESULTS BY PROFITABILITY:")
    print("=" * 80)

    print("\n🏆 TOP 5 PERFORMERS:")
    print("-" * 80)
    for idx, row in results_df.head(5).iterrows():
        print(f"{row['ticker']:6} | Return: {row['return_pct']:+7.2f}% | Trades: {int(row['trades']):3} | "
              f"Win Rate: {row['win_rate']:5.1f}% | Capital: {row['final_capital']:,.0f} RUB")

    print("\n📉 BOTTOM 5 PERFORMERS:")
    print("-" * 80)
    for idx, row in results_df.tail(5).iterrows():
        print(f"{row['ticker']:6} | Return: {row['return_pct']:+7.2f}% | Trades: {int(row['trades']):3} | "
              f"Win Rate: {row['win_rate']:5.1f}% | Capital: {row['final_capital']:,.0f} RUB")

    # Общая статистика
    total_trades = results_df['trades'].sum()
    avg_return = results_df['return_pct'].mean()
    profitable_count = len(results_df[results_df['return_pct'] > 0])

    print("\n" + "=" * 80)
    print("OVERALL STATISTICS:")
    print("=" * 80)
    print(f"Total Instruments Tested: {len(results_df)}")
    print(f"Profitable Instruments: {profitable_count} ({profitable_count/len(results_df)*100:.1f}%)")
    print(f"Average Return: {avg_return:+.2f}%")
    print(f"Best Return: {results_df['return_pct'].max():+.2f}% ({results_df.loc[results_df['return_pct'].idxmax(), 'ticker']})")
    print(f"Worst Return: {results_df['return_pct'].min():+.2f}% ({results_df.loc[results_df['return_pct'].idxmin(), 'ticker']})")
    print(f"Total Trades: {int(total_trades)}")
    print(f"Average Win Rate: {results_df['win_rate'].mean():.2f}%")

    # Портфельная статистика
    portfolio_capital = results_df['final_capital'].sum()
    portfolio_initial = 100000 * len(results_df)
    portfolio_return = ((portfolio_capital - portfolio_initial) / portfolio_initial) * 100

    print("\n" + "=" * 80)
    print("PORTFOLIO SIMULATION (if trading all instruments):")
    print("=" * 80)
    print(f"Initial Capital: {portfolio_initial:,.0f} RUB ({len(results_df)} x 100,000)")
    print(f"Final Capital: {portfolio_capital:,.0f} RUB")
    print(f"Portfolio Return: {portfolio_return:+.2f}%")
    print(f"Total Profit: {portfolio_capital - portfolio_initial:+,.0f} RUB")

    # Рекомендации
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS:")
    print("=" * 80)

    top_performers = results_df[results_df['return_pct'] > 3].head(10)
    if len(top_performers) > 0:
        top_list = top_performers['ticker'].tolist()
        print(f"✅ TOP PERFORMERS (>3% return): {', '.join(top_list)}")
        print(f"   Average return: {top_performers['return_pct'].mean():.2f}%")
        print(f"   Average win rate: {top_performers['win_rate'].mean():.1f}%")

    bad_performers = results_df[results_df['return_pct'] < -1]
    if len(bad_performers) > 0:
        bad_list = bad_performers['ticker'].tolist()
        print(f"\n⚠️  EXCLUDE THESE (<-1% return): {', '.join(bad_list)}")

    moderate = results_df[(results_df['return_pct'] >= -1) & (results_df['return_pct'] <= 3)]
    if len(moderate) > 0:
        print(f"\n📊 MODERATE PERFORMERS (-1% to 3%): {len(moderate)} instruments")

    # Сохраняем результаты
    results_df.to_csv('multi_backtest_results.csv', index=False)
    print("\n📊 Detailed results saved to 'multi_backtest_results.csv'")

    # Создаем рекомендуемый список для config.py
    print("\n" + "=" * 80)
    print("📋 RECOMMENDED INSTRUMENTS FOR config.py:")
    print("=" * 80)
    print("\nINSTRUMENTS = {")
    for ticker in top_performers['ticker'].head(10):
        print(f"    '{ticker}': '{ticker}',")
    print("}")

    return results_df


if __name__ == "__main__":
    # Отключаем графики для многопоточности
    import matplotlib
    matplotlib.use('Agg')

    results = main()