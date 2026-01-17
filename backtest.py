import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

import config
from indicators import TechnicalIndicators
from strategy import TrendStrategy
from trader import TradingBot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """Класс для бэктестинга стратегии на исторических данных"""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.strategy = TrendStrategy()
        self.trades = []
        self.equity_curve = []

    def run_backtest(self, df: pd.DataFrame, ticker: str = "TEST"):
        """Запустить бэктест на исторических данных"""
        logger.info(f"Starting backtest for {ticker}")
        logger.info(f"Data range: {df.index[0]} to {df.index[-1]}")
        logger.info(f"Total candles: {len(df)}")

        # Добавляем индикаторы
        df = TechnicalIndicators.add_all_indicators(df, config)

        position = None  # Текущая открытая позиция

        # Проходим по каждой свече
        for i in range(config.EMA_LONG, len(df)):
            current_price = df.iloc[i]['close']
            current_time = df.index[i]

            # Сохраняем equity
            if position:
                unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
                if position['direction'] == 'sell':
                    unrealized_pnl = -unrealized_pnl
                current_equity = self.capital + unrealized_pnl
            else:
                current_equity = self.capital

            self.equity_curve.append({
                'time': current_time,
                'equity': current_equity
            })

            # Если есть открытая позиция, проверяем условия закрытия
            if position:
                should_close = False
                close_reason = ""

                if position['direction'] == 'buy':
                    if current_price <= position['stop_loss']:
                        should_close = True
                        close_reason = "Stop Loss"
                    elif current_price >= position['take_profit']:
                        should_close = True
                        close_reason = "Take Profit"
                else:  # sell
                    if current_price >= position['stop_loss']:
                        should_close = True
                        close_reason = "Stop Loss"
                    elif current_price <= position['take_profit']:
                        should_close = True
                        close_reason = "Take Profit"

                if should_close:
                    # Закрываем позицию
                    if position['direction'] == 'buy':
                        pnl = (current_price - position['entry_price']) * position['quantity']
                    else:
                        pnl = (position['entry_price'] - current_price) * position['quantity']

                    self.capital += pnl

                    trade_info = {
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'quantity': position['quantity'],
                        'pnl': pnl,
                        'pnl_percent': (pnl / (position['entry_price'] * position['quantity'])) * 100,
                        'reason': close_reason
                    }

                    self.trades.append(trade_info)
                    logger.info(f"Trade closed: {close_reason}, P&L: {pnl:.2f} ({trade_info['pnl_percent']:.2f}%)")

                    position = None

            # Если нет позиции, проверяем сигналы на вход
            if not position:
                # Создаем срез данных до текущей свечи
                df_slice = df.iloc[:i+1]
                signal = self.strategy.get_signal(df_slice)

                if signal['action']:
                    # Рассчитываем размер позиции
                    position_size = self.strategy.calculate_position_size(
                        self.capital, signal['price'], signal['stop_loss']
                    )

                    # Открываем позицию
                    position = {
                        'entry_time': current_time,
                        'entry_price': signal['price'],
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit'],
                        'quantity': position_size,
                        'direction': signal['action'].replace('buy', 'buy').replace('sell', 'sell')
                    }

                    logger.info(f"Position opened: {signal['action']} at {signal['price']:.2f}")

        # Закрываем оставшуюся позицию по последней цене
        if position:
            current_price = df.iloc[-1]['close']
            if position['direction'] == 'buy':
                pnl = (current_price - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - current_price) * position['quantity']

            self.capital += pnl

            self.trades.append({
                'entry_time': position['entry_time'],
                'exit_time': df.index[-1],
                'direction': position['direction'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'quantity': position['quantity'],
                'pnl': pnl,
                'pnl_percent': (pnl / (position['entry_price'] * position['quantity'])) * 100,
                'reason': "End of data"
            })

        self._print_results()
        self._plot_results(df)

    def _print_results(self):
        """Вывести результаты бэктеста"""
        print("\n" + "="*80)
        print("BACKTEST RESULTS")
        print("="*80)

        if not self.trades:
            print("No trades executed")
            return

        trades_df = pd.DataFrame(self.trades)

        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        total_pnl = trades_df['pnl'].sum()
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100

        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

        max_win = trades_df['pnl'].max()
        max_loss = trades_df['pnl'].min()

        print(f"\nInitial Capital: {self.initial_capital:,.2f} RUB")
        print(f"Final Capital: {self.capital:,.2f} RUB")
        print(f"Total Return: {total_return:,.2f}%")
        print(f"Total P&L: {total_pnl:,.2f} RUB")

        print(f"\nTotal Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.2f}%")

        print(f"\nAverage Win: {avg_win:,.2f} RUB")
        print(f"Average Loss: {avg_loss:,.2f} RUB")
        print(f"Max Win: {max_win:,.2f} RUB")
        print(f"Max Loss: {max_loss:,.2f} RUB")

        if avg_loss != 0:
            profit_factor = abs(avg_win / avg_loss)
            print(f"Profit Factor: {profit_factor:.2f}")

        print("\n" + "="*80)

    def _plot_results(self, df: pd.DataFrame):
        """Построить графики результатов"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # График 1: Цена и сделки
        ax1 = axes[0]
        ax1.plot(df.index, df['close'], label='Close Price', alpha=0.7)

        for trade in self.trades:
            if trade['direction'] == 'buy':
                ax1.scatter(trade['entry_time'], trade['entry_price'],
                          color='green', marker='^', s=100, label='Buy Entry')
                ax1.scatter(trade['exit_time'], trade['exit_price'],
                          color='red', marker='v', s=100, label='Buy Exit')
            else:
                ax1.scatter(trade['entry_time'], trade['entry_price'],
                          color='red', marker='v', s=100, label='Sell Entry')
                ax1.scatter(trade['exit_time'], trade['exit_price'],
                          color='green', marker='^', s=100, label='Sell Exit')

        ax1.set_title('Price Chart with Trades')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # График 2: Equity Curve
        ax2 = axes[1]
        equity_df = pd.DataFrame(self.equity_curve)
        ax2.plot(equity_df['time'], equity_df['equity'], label='Equity', color='blue')
        ax2.axhline(y=self.initial_capital, color='gray', linestyle='--', label='Initial Capital')

        ax2.set_title('Equity Curve')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Equity (RUB)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
        print("\n📊 Chart saved as 'backtest_results.png'")
        plt.show()


def main():
    """Запустить бэктест"""
    from tinkoff.invest import Client, CandleInterval

    print("Starting backtest...")

    # Создаем бота для получения данных
    bot = TradingBot(config.TINKOFF_SANDBOX_TOKEN, sandbox=True)

    with Client(config.TINKOFF_SANDBOX_TOKEN) as client:
        ticker_key = config.BACKTEST_TICKER
        ticker_symbol = config.INSTRUMENTS.get(ticker_key)
        if not ticker_symbol:
            available = ", ".join(sorted(config.INSTRUMENTS.keys()))
            print(f"Unknown ticker '{ticker_key}' in BACKTEST_TICKER.")
            print(f"Available tickers: {available}")
            return

        # Получаем FIGI для выбранного тикера
        figi = bot._get_figi_by_ticker(client, ticker_symbol)

        if not figi:
            print("Could not find instrument")
            return

        # Получаем исторические данные (6 месяцев)
        df = bot.get_candles(client, figi, CandleInterval.CANDLE_INTERVAL_HOUR, days=180)

        if df.empty:
            print("No data received")
            return

        # Запускаем бэктест
        backtester = Backtester(initial_capital=100000)
        backtester.run_backtest(df, ticker=ticker_key)


if __name__ == "__main__":
    main()
