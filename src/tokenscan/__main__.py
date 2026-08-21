"""TokenScan — punto de entrada CLI.

Comandos:
  python -m tokenscan run          # bucle del agente (paper/live según config)
  python -m tokenscan telegram     # arranca el bot de Telegram
  python -m tokenscan backtest     # ejecuta backtest de la estrategia configurada
  python -m tokenscan wallet       # muestra wallet / PnL
"""

from __future__ import annotations

import argparse
import sys
import time

from .agent.agent import LLMAgent
from .backtest.engine import run_backtest
from .config import Settings
from .data.market import MarketData
from .execution.exchange import ExchangeClient
from .execution.paper import PaperBroker
from .quant.strategies import get_strategy
from .storage.db import Database
from .utils.logger import setup_logger

log = setup_logger("tokenscan.main")


def build_context(settings: Settings, with_exchange: bool | None = None):
    db = Database(settings.data_dir.rstrip("/") + "/tokenscan.db")
    broker = PaperBroker(settings, db)
    broker.ensure_wallet()
    want_exchange = with_exchange if with_exchange is not None else False
    exchange = ExchangeClient(settings) if want_exchange else None
    market = MarketData(exchange)
    agent = LLMAgent(settings, broker, db, market)
    return db, broker, market, agent


def cmd_run(settings: Settings) -> None:
    _db, _broker, _market, agent = build_context(settings, with_exchange=True)
    log.info("TokenScan arrancando en modo %s", settings.mode)
    log.info("Agente: %s", "LLM (" + settings.llm_model + ")" if agent.available else "determinista (RSI)")
    try:
        while True:
            agent.run_cycle()
            time.sleep(settings.interval)
    except KeyboardInterrupt:
        log.info("Detenido por el usuario.")


def cmd_telegram(settings: Settings) -> None:
    if not settings.telegram_bot_token or settings.telegram_bot_token.startswith("123456789"):
        log.error("TELEGRAM_BOT_TOKEN no configurado. Copia .env.example a .env y pon el token de @BotFather")
        sys.exit(1)
    db, broker, _market, agent = build_context(settings, with_exchange=True)
    from .telegram.bot import run_telegram
    app = run_telegram(settings, db, broker, agent)
    log.info("Bot de Telegram arrancando...")
    app.run_polling()


def cmd_backtest(settings: Settings) -> None:
    _db, _broker, market, _agent = build_context(settings, with_exchange=True)
    strategy = get_strategy(
        settings.strategy.name,
        rsi_period=settings.strategy.rsi_period,
        oversold=settings.strategy.rsi_oversold,
        overbought=settings.strategy.rsi_overbought,
    )
    log.info("Backtest %s sobre %d días (%s)", strategy.name, settings.backtest.days, settings.timeframe)
    result = run_backtest(settings, market, strategy)
    print("\n===== RESULTADO BACKTEST =====")
    for k, v in result.metrics().items():
        print(f"  {k}: {v}")
    print("=============================\n")
    if result.trades:
        first = result.trades[0]
        print(f"  {len(result.trades)} trades. Ejemplo: {first.pair} {first.side} "
              f"{first.open_price:.2f}->{first.close_price:.2f} ({first.exit_reason})")


def cmd_wallet(settings: Settings) -> None:
    _db, broker, _market, _agent = build_context(settings)
    snap = broker.to_snapshot()
    print("=== WALLET TokenScan ===")
    print(f"  Balance: {snap['balance']:.2f} USDT")
    print(f"  Equity:  {snap['equity']:.2f} USDT")
    print(f"  Posiciones: {snap['positions']}")
    print(f"  Stats:   {snap['stats']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tokenscan", description="TokenScan — AI trading agent para cripto")
    parser.add_argument("command", nargs="?", default="run",
                        choices=["run", "telegram", "backtest", "wallet"])
    parser.add_argument("--config", help="ruta alternativa a config.yaml")
    args = parser.parse_args(argv)

    settings = Settings.load(args.config)
    log.info("Config: modo=%s timeframe=%s pares=%s", settings.mode, settings.timeframe, settings.trading_pairs)

    if args.command == "run":
        cmd_run(settings)
    elif args.command == "telegram":
        cmd_telegram(settings)
    elif args.command == "backtest":
        cmd_backtest(settings)
    elif args.command == "wallet":
        cmd_wallet(settings)


if __name__ == "__main__":
    main()
