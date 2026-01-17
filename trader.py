import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from tinkoff.invest import Client, CandleInterval
from tinkoff.invest.utils import now

import config
from indicators import TechnicalIndicators
from strategy import TrendStrategy


# Настройка логирования
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TradingBot:
    """Основной класс торгового бота"""

    def __init__(self, token: str, sandbox: bool = True):
        self.token = token
        self.sandbox = sandbox
        self.strategy = TrendStrategy()
        self.positions = {}  # Открытые позиции
        self.account_id = None

    def _get_account_id(self, client: Client) -> str:
        """Получить ID реального счета"""
        accounts = client.users.get_accounts()
        if not accounts.accounts:
            raise ValueError("No accounts found")

        account = accounts.accounts[0]
        logger.info(f"Using account: {account.id} (Type: {account.type})")
        return account.id

    def _get_sandbox_account(self, client: Client) -> str:
        """Получить или создать sandbox аккаунт"""
        try:
            # Пытаемся получить существующие аккаунты
            accounts = client.sandbox.get_sandbox_accounts()

            if accounts.accounts:
                account = accounts.accounts[0]
                logger.info(f"Using existing sandbox account: {account.id}")
                return account.id
            else:
                # Создаем новый sandbox аккаунт
                response = client.sandbox.open_sandbox_account()
                logger.info(f"Created new sandbox account: {response.account_id}")
                return response.account_id

        except Exception as e:
            logger.error(f"Error getting sandbox account: {e}")
            raise

    def _get_figi_by_ticker(self, client: Client, ticker: str) -> Optional[str]:
        """Получить FIGI по тикеру"""
        try:
            # Если уже передан FIGI
            if ticker.startswith('FUT') or len(ticker) > 10:
                logger.info(f"Using provided FIGI: {ticker}")
                return ticker

            # Ищем инструмент
            instruments = client.instruments.find_instrument(query=ticker)

            # Ищем ближайший фьючерс (по дате экспирации)
            futures = [inst for inst in instruments.instruments
                       if hasattr(inst, 'expiration_date') and inst.ticker.startswith(ticker)]

            if futures:
                # Сортируем по дате экспирации и берем ближайший
                from datetime import datetime
                futures_sorted = sorted(futures,
                                       key=lambda x: x.expiration_date if hasattr(x, 'expiration_date') else datetime.max)

                nearest = futures_sorted[0]
                logger.info(f"Found nearest future: {nearest.name} (FIGI: {nearest.figi})")
                return nearest.figi

            # Если не нашли фьючерсы, берем любой подходящий инструмент
            for instrument in instruments.instruments:
                if ticker.upper() in instrument.ticker.upper():
                    logger.info(f"Found instrument: {instrument.name} (FIGI: {instrument.figi})")
                    return instrument.figi

            logger.warning(f"Instrument {ticker} not found")
            return None
        except Exception as e:
            logger.error(f"Error finding instrument {ticker}: {e}")
            return None

    def get_candles(self, client: Client, figi: str, interval: CandleInterval,
                    days: int = 30) -> pd.DataFrame:
        """Получить исторические свечи"""
        try:
            # Временной диапазон
            to_time = now()
            from_time = to_time - timedelta(days=days)

            logger.info(f"Fetching candles from {from_time} to {to_time}")

            # Получаем свечи
            candles = []
            for candle in client.get_all_candles(
                figi=figi,
                from_=from_time,
                to=to_time,
                interval=interval
            ):
                candles.append(candle)

            if not candles:
                logger.warning("No candles received")
                return pd.DataFrame()

            # Конвертируем в DataFrame
            df = pd.DataFrame([{
                'time': candle.time,
                'open': self._quotation_to_float(candle.open),
                'high': self._quotation_to_float(candle.high),
                'low': self._quotation_to_float(candle.low),
                'close': self._quotation_to_float(candle.close),
                'volume': candle.volume
            } for candle in candles])

            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)

            logger.info(f"Loaded {len(df)} candles")
            return df

        except Exception as e:
            logger.error(f"Error fetching candles: {e}")
            return pd.DataFrame()

    @staticmethod
    def _quotation_to_float(quotation) -> float:
        """Конвертировать Quotation в float"""
        if quotation is None:
            return 0.0
        return quotation.units + quotation.nano / 1e9

    @staticmethod
    def _float_to_quotation(value: float):
        """Конвертировать float в Quotation"""
        from tinkoff.invest import Quotation, MoneyValue
        units = int(value)
        nano = int((value - units) * 1e9)
        return MoneyValue(currency='rub', units=units, nano=nano)

    def get_portfolio(self, client: Client) -> Dict:
        """Получить информацию о портфеле"""
        try:
            if self.sandbox:
                portfolio = client.sandbox.get_sandbox_portfolio(account_id=self.account_id)
            else:
                portfolio = client.operations.get_portfolio(account_id=self.account_id)

            total_amount = self._quotation_to_float(portfolio.total_amount_portfolio)

            positions = []
            for position in portfolio.positions:
                positions.append({
                    'figi': position.figi,
                    'quantity': self._quotation_to_float(position.quantity),
                    'current_price': self._quotation_to_float(position.current_price),
                    'average_price': self._quotation_to_float(position.average_position_price),
                })

            return {
                'total_amount': total_amount,
                'positions': positions
            }

        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            return {'total_amount': 0, 'positions': []}

    def place_order(self, client: Client, figi: str, quantity: int,
                    direction: str, price: Optional[float] = None) -> bool:
        """Разместить ордер"""
        try:
            from tinkoff.invest import OrderDirection, OrderType
            import uuid

            # Определяем направление
            order_direction = (OrderDirection.ORDER_DIRECTION_BUY
                             if direction == 'buy'
                             else OrderDirection.ORDER_DIRECTION_SELL)

            logger.info(f"Placing {direction} order: {quantity} lots at {price if price else 'market'}")

            # Генерируем уникальный ID для ордера
            order_id = str(uuid.uuid4())

            if self.sandbox:
                # В песочнице используем sandbox метод
                response = client.sandbox.post_sandbox_order(
                    account_id=self.account_id,
                    figi=figi,
                    quantity=quantity,
                    direction=order_direction,
                    order_type=OrderType.ORDER_TYPE_MARKET,
                    order_id=order_id
                )
            else:
                # На реальном счете
                response = client.orders.post_order(
                    account_id=self.account_id,
                    figi=figi,
                    quantity=quantity,
                    direction=order_direction,
                    order_type=OrderType.ORDER_TYPE_MARKET,
                    order_id=order_id
                )

            logger.info(f"Order placed successfully: {response.order_id}")
            return True

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return False

    def check_and_execute_signals(self, client: Client, ticker: str, figi: str):
        """Проверить сигналы и исполнить сделки"""
        try:
            # Получаем данные
            df = self.get_candles(client, figi, CandleInterval.CANDLE_INTERVAL_HOUR, days=365)

            if df.empty or len(df) < config.EMA_LONG:
                logger.warning(f"Not enough data for {ticker}")
                return

            # Добавляем индикаторы
            df = TechnicalIndicators.add_all_indicators(df, config)

            # Получаем сигнал
            signal = self.strategy.get_signal(df)

            if signal['action'] is None:
                logger.info(f"No signal for {ticker}")
                return

            logger.info(f"Signal for {ticker}: {signal['action']} at {signal['price']}")
            logger.info(f"Reason: {signal['reason']}")
            logger.info(f"Stop Loss: {signal['stop_loss']:.2f}, Take Profit: {signal['take_profit']:.2f}")

            # Проверяем портфель
            portfolio = self.get_portfolio(client)
            capital = portfolio['total_amount']

            if capital <= 0:
                logger.warning("No capital available")
                return

            # Рассчитываем размер позиции
            position_size = self.strategy.calculate_position_size(
                capital, signal['price'], signal['stop_loss']
            )

            logger.info(f"Position size: {position_size} lots")

            # Проверяем лимит позиций
            if len(self.positions) >= config.MAX_POSITIONS:
                logger.warning(f"Maximum positions reached ({config.MAX_POSITIONS})")
                return

            # Размещаем ордер
            success = self.place_order(
                client, figi, position_size,
                'buy' if signal['action'] == 'buy' else 'sell'
            )

            if success:
                # Сохраняем информацию о позиции
                self.positions[ticker] = {
                    'figi': figi,
                    'direction': signal['action'],
                    'entry_price': signal['price'],
                    'stop_loss': signal['stop_loss'],
                    'take_profit': signal['take_profit'],
                    'quantity': position_size,
                    'opened_at': datetime.now()
                }
                logger.info(f"Position opened for {ticker}")

        except Exception as e:
            logger.error(f"Error checking signals for {ticker}: {e}")

    def check_positions(self, client: Client):
        """Проверить открытые позиции и закрыть при необходимости"""
        for ticker, position in list(self.positions.items()):
            try:
                # Получаем текущую цену
                df = self.get_candles(
                    client, position['figi'],
                    CandleInterval.CANDLE_INTERVAL_1_MIN,
                    days=1
                )

                if df.empty:
                    continue

                current_price = df.iloc[-1]['close']

                logger.info(f"Checking position {ticker}: current={current_price:.2f}, "
                          f"entry={position['entry_price']:.2f}, "
                          f"SL={position['stop_loss']:.2f}, TP={position['take_profit']:.2f}")

                # Проверяем стоп-лосс и тейк-профит
                should_close = False
                reason = ""

                if position['direction'] == 'buy':
                    if current_price <= position['stop_loss']:
                        should_close = True
                        reason = "Stop Loss hit"
                    elif current_price >= position['take_profit']:
                        should_close = True
                        reason = "Take Profit hit"
                else:  # sell
                    if current_price >= position['stop_loss']:
                        should_close = True
                        reason = "Stop Loss hit"
                    elif current_price <= position['take_profit']:
                        should_close = True
                        reason = "Take Profit hit"

                if should_close:
                    logger.info(f"Closing position {ticker}: {reason}")

                    # Закрываем позицию (противоположный ордер)
                    close_direction = 'sell' if position['direction'] == 'buy' else 'buy'
                    success = self.place_order(
                        client, position['figi'],
                        position['quantity'], close_direction
                    )

                    if success:
                        # Рассчитываем P&L
                        if position['direction'] == 'buy':
                            pnl = (current_price - position['entry_price']) * position['quantity']
                        else:
                            pnl = (position['entry_price'] - current_price) * position['quantity']

                        logger.info(f"Position closed. P&L: {pnl:.2f} RUB")
                        del self.positions[ticker]

            except Exception as e:
                logger.error(f"Error checking position {ticker}: {e}")

    def run(self):
        """Запустить бота"""
        logger.info("=" * 80)
        logger.info(f"Starting Trading Bot - {self.strategy.name}")
        logger.info(f"Mode: {'SANDBOX' if self.sandbox else 'REAL'}")
        logger.info(f"Instruments: {list(config.INSTRUMENTS.keys())}")
        logger.info("=" * 80)

        try:
            with Client(self.token) as client:
                # Получаем ID счета (разные методы для sandbox и real)
                if self.sandbox:
                    self.account_id = self._get_sandbox_account(client)

                    # Пополняем sandbox счет виртуальными деньгами
                    try:
                        client.sandbox.sandbox_pay_in(
                            account_id=self.account_id,
                            amount=self._float_to_quotation(config.INITIAL_CAPITAL)
                        )
                        logger.info(f"Sandbox account funded with {config.INITIAL_CAPITAL} RUB")
                    except Exception as e:
                        logger.warning(f"Could not fund sandbox account: {e}")
                else:
                    self.account_id = self._get_account_id(client)

                # Получаем портфель
                portfolio = self.get_portfolio(client)
                logger.info(f"Portfolio value: {portfolio['total_amount']:.2f} RUB")

                # Основной цикл
                for ticker in config.INSTRUMENTS.keys():
                    logger.info(f"\nAnalyzing {ticker}...")

                    # Получаем FIGI
                    figi = self._get_figi_by_ticker(client, config.INSTRUMENTS[ticker])

                    if not figi:
                        logger.warning(f"Could not find FIGI for {ticker}")
                        continue

                    # Проверяем сигналы
                    self.check_and_execute_signals(client, ticker, figi)

                # Проверяем открытые позиции
                if self.positions:
                    logger.info("\nChecking open positions...")
                    self.check_positions(client)

                logger.info("\n" + "=" * 80)
                logger.info("Bot cycle completed")
                logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Error running bot: {e}", exc_info=True)


if __name__ == "__main__":
    # Тестовый запуск
    bot = TradingBot(config.TINKOFF_SANDBOX_TOKEN, sandbox=True)
    bot.run()