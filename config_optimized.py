import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
TINKOFF_SANDBOX_TOKEN = os.getenv('TINKOFF_SANDBOX_TOKEN')
TINKOFF_REAL_TOKEN = os.getenv('TINKOFF_REAL_TOKEN') or os.getenv('TINKOFF_READONLY_TOKEN')
TINKOFF_READONLY_TOKEN = os.getenv('TINKOFF_READONLY_TOKEN')

# Trading Configuration
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 100000))
RISK_PER_TRADE = 0.025  # УВЕЛИЧЕНО с 0.02 до 0.025 (2.5%)
MAX_POSITIONS = 5       # Можем держать 5 позиций одновременно

# Strategy Parameters - АГРЕССИВНЫЕ
TIMEFRAME = 'day'
EMA_SHORT = 20
EMA_MEDIUM = 50
EMA_LONG = 200

RSI_PERIOD = 14
RSI_OVERSOLD = 33
RSI_OVERBOUGHT = 67

ADX_PERIOD = 14
ADX_THRESHOLD = 28  # Снижено с 30 для большего количества сигналов

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Instruments - БУДЕТ ОБНОВЛЕНО после теста
# Оставим только ТОП-10 самых прибыльных
INSTRUMENTS = {
    'SBER': 'SBER',
    'GAZP': 'GAZP',
    'LKOH': 'LKOH',
    'GMKN': 'GMKN',
    'YNDX': 'YNDX',
    'VTBR': 'VTBR',
    'ROSN': 'ROSN',
    'NVTK': 'NVTK',
    'NLMK': 'NLMK',
    'MTSS': 'MTSS',
}

# Risk Management - АГРЕССИВНЫЕ
STOP_LOSS_PERCENT = 0.022  # 2.2%
TAKE_PROFIT_RATIO = 3.0    # Risk/Reward 1:3 (было 2.5)
TRAILING_STOP = True
TRAILING_STOP_PERCENT = 0.015

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'trading_bot.log'
```

---

## После завершения теста:

Вы увидите что-то вроде:
```
TOP 5 PERFORMERS:
GMKN   | Return: +8.5% | Trades: 25 | Win Rate: 52% | ...
LKOH   | Return: +6.2% | Trades: 22 | Win Rate: 48% | ...
SBER   | Return: +5.1% | Trades: 20 | Win Rate: 45% | ...
...

PORTFOLIO SIMULATION:
Initial Capital: 2,100,000 RUB
Final Capital: 2,250,000 RUB
Portfolio Return: +7.14%
Total Profit: +150,000 RUB