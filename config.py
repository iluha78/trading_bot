import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
TINKOFF_SANDBOX_TOKEN = os.getenv('TINKOFF_SANDBOX_TOKEN')
TINKOFF_REAL_TOKEN = os.getenv('TINKOFF_REAL_TOKEN') or os.getenv('TINKOFF_READONLY_TOKEN')
TINKOFF_READONLY_TOKEN = os.getenv('TINKOFF_READONLY_TOKEN')

# Trading Configuration
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 100000))
RISK_PER_TRADE = 0.015  # 1.5% на сделку (меньше для частой торговли)
MAX_POSITIONS = 3       # Максимум 3 позиции одновременно

# Strategy Parameters - ДЛЯ МИНУТНОЙ ТОРГОВЛИ
TIMEFRAME = '15min'
EMA_SHORT = 10      # Короткие периоды для минутных свечей
EMA_MEDIUM = 15    # 5 минута
EMA_LONG = 20      # 50 минут

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

ADX_PERIOD = 14
ADX_THRESHOLD = 20  # Ниже для минутных свечей

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Instruments - ТОП-7 лучших
INSTRUMENTS = {
    # Банки
    'SBER': 'SBER',     # Сбербанк - голубая фишка
    'VTBR': 'VTBR',     # ВТБ

    # Нефть и газ
    'GAZP': 'GAZP',     # Газпром
    'LKOH': 'LKOH',     # Лукойл
    'ROSN': 'ROSN',     # Роснефть
    'NVTK': 'NVTK',     # Новатэк
    'TATN': 'TATN',     # Татнефть

    # Металлургия
    'GMKN': 'GMKN',     # ГМК Норникель
    'NLMK': 'NLMK',     # НЛМК
    'MAGN': 'MAGN',     # ММК
    'CHMF': 'CHMF',     # Северсталь

    # Технологии
    'YNDX': 'YNDX',     # Яндекс
    'VKCO': 'VKCO',     # VK

    # Ритейл и потребительский сектор
    'MGNT': 'MGNT',     # Магнит
    'FIVE': 'FIVE',     # X5 Retail Group
    'AFKS': 'AFKS',     # Система АФК

    # Телеком
    'MTSS': 'MTSS',     # МТС
    'RTKM': 'RTKM',     # Ростелеком

    # Химия и удобрения
    'PHOR': 'PHOR',     # ФосАгро

    # Энергетика
    'FEES': 'FEES',     # ФСК ЕЭС
    'HYDR': 'HYDR',     # РусГидро
}

# Risk Management - ДЛЯ СКАЛЬПИНГА
STOP_LOSS_PERCENT = 0.008   # 0.8% стоп (узкий для минутной торговли)
TAKE_PROFIT_RATIO = 2.0     # R/R 1:2 (быстрый профит)
TRAILING_STOP = True
TRAILING_STOP_PERCENT = 0.005  # 0.5% трейлинг

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'trading_bot.log'