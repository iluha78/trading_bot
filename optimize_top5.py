import pandas as pd
from itertools import product
from backtest import Backtester
from demo_backtest import generate_synthetic_data
import config

def optimize_top5():
    """Оптимизация параметров для ТОП-5 акций"""

    top_tickers = ['SBER', 'HYDR', 'CHMF', 'TATN', 'NVTK']

    # Параметры для тестирования
    adx_values = [25, 28, 30]
    rr_values = [2.5, 3.0, 3.5, 4.0]
    risk_values = [0.02, 0.025, 0.03, 0.035]

    best_results = []

    print("Optimizing parameters for TOP-5 instruments...")
    print(f"Total combinations: {len(list(product(adx_values, rr_values, risk_values)))}\n")

    for ticker in top_tickers:
        print(f"\n{'='*60}")
        print(f"Optimizing {ticker}...")
        print('='*60)

        ticker_results = []

        for adx, rr, risk in product(adx_values, rr_values, risk_values):
            # Временно меняем параметры
            original_adx = config.ADX_THRESHOLD
            original_rr = config.TAKE_PROFIT_RATIO
            original_risk = config.RISK_PER_TRADE

            config.ADX_THRESHOLD = adx
            config.TAKE_PROFIT_RATIO = rr
            config.RISK_PER_TRADE = risk

            # Генерируем данные
            df = generate_synthetic_data(days=365, volatility=0.015)

            # Бэктест без вывода
            import sys, io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            backtester = Backtester(initial_capital=100000)

            # Перезагружаем стратегию
            import importlib
            import strategy
            importlib.reload(strategy)
            backtester.strategy = strategy.TrendStrategy()

            try:
                backtester.run_backtest(df, ticker=ticker)
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
                return_pct = ((backtester.capital - 100000) / 100000) * 100
                win_rate = (len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)) * 100

                ticker_results.append({
                    'ADX': adx,
                    'RR': rr,
                    'Risk%': risk * 100,
                    'Return%': return_pct,
                    'Trades': len(trades_df),
                    'WinRate%': win_rate
                })

        # Лучшая конфигурация для акции
        if ticker_results:
            results_df = pd.DataFrame(ticker_results)
            best = results_df.loc[results_df['Return%'].idxmax()]

            print(f"\n🏆 Best for {ticker}:")
            print(f"   ADX: {best['ADX']}, R/R: {best['RR']}, Risk: {best['Risk%']:.1f}%")
            print(f"   Return: {best['Return%']:+.2f}%, Win Rate: {best['WinRate%']:.1f}%")

            best_results.append({
                'Ticker': ticker,
                'ADX': best['ADX'],
                'RR': best['RR'],
                'Risk%': best['Risk%'],
                'Return%': best['Return%']
            })

    # Общие рекомендации
    print("\n" + "="*80)
    print("RECOMMENDED PARAMETERS (average of best results):")
    print("="*80)

    best_df = pd.DataFrame(best_results)
    print(f"\nADX_THRESHOLD: {int(best_df['ADX'].mean())}")
    print(f"TAKE_PROFIT_RATIO: {best_df['RR'].mean():.1f}")
    print(f"RISK_PER_TRADE: {best_df['Risk%'].mean() / 100:.3f}")
    print(f"\nExpected Return: {best_df['Return%'].mean():+.2f}%")

    return best_df

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')

    results = optimize_top5()