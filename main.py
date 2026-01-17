import time
import logging
from datetime import datetime
import config
from trader import TradingBot

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Главная функция для запуска бота"""

    print("=" * 80)
    print("TRADING BOT - ST Sentiment Trading Strategy")
    print("=" * 80)
    print("\nSelect mode:")
    print("1. Sandbox (Test mode with virtual money)")
    print("2. Real (Live trading with real money)")
    print("3. Single run (Test once)")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == '1':
        sandbox = True
        token = config.TINKOFF_SANDBOX_TOKEN
        continuous = True
        print("\n✅ Running in SANDBOX mode")
    elif choice == '2':
        sandbox = False
        token = config.TINKOFF_REAL_TOKEN
        continuous = True

        print("\n⚠️  WARNING: This will use REAL MONEY!")
        confirm = input("Type 'YES' to confirm: ").strip()

        if confirm != 'YES':
            print("Aborted.")
            return

        print("\n✅ Running in REAL mode")
    elif choice == '3':
        sandbox = True
        token = config.TINKOFF_SANDBOX_TOKEN
        continuous = False
        print("\n✅ Single run in SANDBOX mode")
    else:
        print("Invalid choice")
        return

    if not token:
        logger.error("Token not found! Please check your .env file")
        return

    # Создаем бота
    bot = TradingBot(token, sandbox=sandbox)

    if continuous:
        print(f"\nBot will run continuously (check every 1 hour)")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                logger.info(f"\n{'='*80}")
                logger.info(f"Bot cycle started at {datetime.now()}")
                logger.info(f"{'='*80}\n")

                bot.run()

                # Ждем 1 час до следующего запуска
                logger.info("\nWaiting 1 hour until next cycle...")
                time.sleep(3600)  # 3600 секунд = 1 час

        except KeyboardInterrupt:
            logger.info("\n\nBot stopped by user")
            print("\n✅ Bot stopped successfully")

    else:
        # Однократный запуск
        bot.run()
        print("\n✅ Single run completed")


if __name__ == "__main__":
    main()