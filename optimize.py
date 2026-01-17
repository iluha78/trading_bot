import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
import matplotlib.pyplot as plt

from backtest import Backtester
from demo_backtest import generate_synthetic_data
import config


def optimize_parameters():
    """Оптимизация параметров стратегии"""

    print("=" * 80)
    print("STRATEGY OPTIMIZATION")
    print("=" * 80)

    # Генерируем данные для теста
    df = generate_synthetic_data(days=365, volatility=0.015)

    # Параметры для тестирования
    adx_thresholds = [25, 28, 30, 32, 35]
    risk_rewards = [2.0, 2.5, 3.0, 3.5]
    risk_per_trades = [0.01, 0.02, 0.03, 0.04]

    results = []
    total_combinations = len(adx_thresholds) * len(risk_rewards) * len(risk_per_trades)
    current = 0

    print(f"\nTesting {total_combinations} combinations...\n")

    for adx, rr, risk in product(adx_thresholds, risk_rewards, risk_per_trades):
        current += 1

        # Временно меняем параметры
        original_adx = config.ADX_THRESHOLD
        original_rr = config.TAKE_PROFIT_RATIO
        original_risk = config.RISK_PER_TRADE

        config.ADX_THRESHOLD = adx
        config.TAKE_PROFIT_RATIO = rr
        config.RISK_PER_TRADE = risk

        # Запускаем бэктест
        backtester = Backtester(initial_capital=100000)

        # Перезагружаем модули чтобы применить новые настройки
        import importlib
        import strategy
        importlib.reload(strategy)
        backtester.strategy = strategy.TrendStrategy()

        # Запускаем без вывода
        import sys
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            backtester.run_backtest(df, ticker='OPTIMIZE')
        except:
            pass

        sys.stdout = old_stdout

        # Восстанавливаем параметры
        config.ADX_THRESHOLD = original_adx
        config.TAKE_PROFIT_RATIO = original_rr
        config.RISK_PER_TRADE = original_risk

        # Сохраняем результаты
        if backtester.trades:
            trades_df = pd.DataFrame(backtester.trades)
            total_return = ((backtester.capital - backtester.initial_capital) /
                          backtester.initial_capital) * 100
            win_rate = (len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)) * 100

            results.append({
                'ADX': adx,
                'Risk/Reward': rr,
                'Risk%': risk * 100,
                'Return%': total_return,
                'Trades': len(trades_df),
                'Win Rate%': win_rate,
                'Final Capital': backtester.capital
            })

            if current % 10 == 0:
                print(f"Progress: {current}/{total_combinations} ({current/total_combinations*100:.1f}%)")

    # Анализ результатов
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Return%', ascending=False)

    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS:")
    print("=" * 80)
    print(results_df.head(10).to_string(index=False))

    # Лучшая конфигурация
    best = results_df.iloc[0]

    print("\n" + "=" * 80)
    print("🏆 BEST CONFIGURATION:")
    print("=" * 80)
    print(f"ADX Threshold: {best['ADX']}")
    print(f"Risk/Reward: {best['Risk/Reward']}")
    print(f"Risk per Trade: {best['Risk%']:.1f}%")
    print(f"Expected Return: {best['Return%']:.2f}%")
    print(f"Win Rate: {best['Win Rate%']:.2f}%")
    print(f"Total Trades: {int(best['Trades'])}")

    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # График 1: Return% vs ADX
    results_df.groupby('ADX')['Return%'].mean().plot(ax=axes[0, 0], marker='o')
    axes[0, 0].set_title('Return vs ADX Threshold')
    axes[0, 0].set_xlabel('ADX Threshold')
    axes[0, 0].set_ylabel('Average Return %')
    axes[0, 0].grid(True)

    # График 2: Return% vs Risk/Reward
    results_df.groupby('Risk/Reward')['Return%'].mean().plot(ax=axes[0, 1], marker='o')
    axes[0, 1].set_title('Return vs Risk/Reward Ratio')
    axes[0, 1].set_xlabel('Risk/Reward')
    axes[0, 1].set_ylabel('Average Return %')
    axes[0, 1].grid(True)

    # График 3: Return% vs Risk per Trade
    results_df.groupby('Risk%')['Return%'].mean().plot(ax=axes[1, 0], marker='o')
    axes[1, 0].set_title('Return vs Risk per Trade')
    axes[1, 0].set_xlabel('Risk per Trade %')
    axes[1, 0].set_ylabel('Average Return %')
    axes[1, 0].grid(True)

    # График 4: Win Rate vs Trades
    axes[1, 1].scatter(results_df['Trades'], results_df['Win Rate%'],
                       c=results_df['Return%'], cmap='RdYlGn', alpha=0.6)
    axes[1, 1].set_title('Win Rate vs Number of Trades')
    axes[1, 1].set_xlabel('Number of Trades')
    axes[1, 1].set_ylabel('Win Rate %')
    axes[1, 1].grid(True)

    plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Return %')
    plt.tight_layout()
    plt.savefig('optimization_results.png', dpi=300)
    print("\n📊 Optimization charts saved as 'optimization_results.png'")

    return best


if __name__ == "__main__":
    best_config = optimize_parameters()