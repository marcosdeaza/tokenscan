"""Bot de Telegram de TokenScan: control de wallet, órdenes, PnL y del agente."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from ..agent.agent import LLMAgent
from ..config import Settings
from ..execution.paper import PaperBroker
from ..storage.db import Database
from ..utils.logger import setup_logger
from ..wallet import create_wallet, wallet_factory

log = setup_logger("tokenscan.telegram")

Handler = Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[Any]]


def authorized_only(fn: Handler) -> Handler:
    @wraps(fn)
    async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> Any:
        chat_id = str(update.effective_chat.id) if update.effective_chat else None
        if chat_id != str(self.s.telegram_chat_id):
            await update.message.reply_text("⛔ No autorizado.")
            return None
        return await fn(self, update, context)
    return wrapper


class TokenScanBot:
    def __init__(self, settings: Settings, db: Database, broker: PaperBroker, agent: LLMAgent):
        self.s = settings
        self.db = db
        self.broker = broker
        self.agent = agent
        self._agent_running = False
        self._agent_task: asyncio.Task | None = None

    def start_agent(self) -> None:
        if self._agent_running:
            return
        self._agent_running = True
        self._agent_task = asyncio.get_event_loop().create_task(self._agent_loop())
        log.info("[BOT] Agente iniciado (intervalo %ds)", self.s.interval)

    def stop_agent(self) -> None:
        self._agent_running = False
        if self._agent_task:
            self._agent_task.cancel()
            self._agent_task = None
        log.info("[BOT] Agente detenido")

    async def _agent_loop(self) -> None:
        while self._agent_running:
            try:
                decisions = self.agent.run_cycle()
                if self.s.telegram.notify_cycles and decisions:
                    await self._notify_cycles(decisions)
            except Exception as e:  # noqa: BLE001
                log.error("[BOT] Error en ciclo agente: %s", e)
            await asyncio.sleep(self.s.interval)

    async def _notify_cycles(self, decisions: list) -> None:
        lines = [f"🔄 Ciclo agente: {len(decisions)} operaciones"]
        for d in decisions:
            lines.append(f"  • {d.action.upper()} {d.pair} (conf {d.confidence:.0f})")
        await self._send("\n".join(lines))

    async def _send(self, text: str) -> None:
        from telegram import Bot
        bot = Bot(self.s.telegram_bot_token)
        await bot.send_message(chat_id=self.s.telegram_chat_id, text=text, parse_mode=ParseMode.HTML)

    # ── Handlers ──────────────────────────────────────────────

    @authorized_only
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "👋 <b>TokenScan</b> — agente de trading IA para cripto.\n\n"
            "Comandos:\n"
            "/wallet — ver mi wallet\n"
            "/deposit &lt;cantidad&gt; — ingresar (paper)\n"
            "/withdraw &lt;cantidad&gt; — retirar (paper)\n"
            "/balance — saldo y equity\n"
            "/positions — posiciones abiertas\n"
            "/pnl — estadísticas de rendimiento\n"
            "/trade &lt;PAR&gt; &lt;buy|sell&gt; &lt;monto&gt; — operación manual\n"
            "/agent_start / agent_stop — arrancar/parar el agente IA\n"
            "/wallet_onchain — ver cartera blockchain real\n"
            "/create_wallet &lt;base|solana&gt; — crear cartera real\n"
            "/status — estado del sistema",
            parse_mode=ParseMode.HTML,
        )

    @authorized_only
    async def cmd_wallet(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        w = self.db.get_wallet(self.broker.wallet_id)
        if self.s.mode == "live":
            from ..execution.live import LiveBroker
            if isinstance(self.broker, LiveBroker):
                usdc = self.broker.get_balance()
                sol = self.broker.jupiter.get_sol_balance()
                await update.message.reply_text(
                    f"💼 Wallet <b>{w['label']}</b> (on-chain)\n"
                    f"  • Dirección: <code>{self.broker.jupiter.address}</code>\n"
                    f"  • USDC: {usdc:.4f}\n"
                    f"  • SOL: {sol:.6f}\n"
                    f"  • Equity: {self.broker.get_equity():.2f} USDC\n"
                    f"  • Posiciones: {len(self.broker.positions)}",
                    parse_mode=ParseMode.HTML,
                )
                return
        await update.message.reply_text(
            f"💼 Wallet <b>{w['label']}</b>\n"
            f"  • Moneda: {w['currency']}\n"
            f"  • Saldo: {w['balance']:.2f} {w['currency']}\n"
            f"  • Equity: {self.broker.get_equity():.2f}\n"
            f"  • Posiciones: {len(self.broker.positions)}",
            parse_mode=ParseMode.HTML,
        )

    @authorized_only
    async def cmd_deposit(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        amount = float(context.args[0]) if context.args else 0.0
        try:
            balance = self.broker.deposit(amount)
            await update.message.reply_text(f"✅ Ingresados {amount:.2f} USDT. Saldo: {balance:.2f}")
        except (ValueError, IndexError) as e:
            await update.message.reply_text(f"❌ {e}")

    @authorized_only
    async def cmd_withdraw(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        amount = float(context.args[0]) if context.args else 0.0
        try:
            balance = self.broker.withdraw(amount)
            await update.message.reply_text(f"✅ Retirados {amount:.2f} USDT. Saldo: {balance:.2f}")
        except (ValueError, IndexError) as e:
            await update.message.reply_text(f"❌ {e}")

    @authorized_only
    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        snap = self.broker.to_snapshot()
        if self.s.mode == "live":
            from ..execution.live import LiveBroker
            if isinstance(self.broker, LiveBroker):
                sol = self.broker.jupiter.get_sol_balance()
                await update.message.reply_text(
                    f"💰 <b>Balance (on-chain)</b>\n"
                    f"  • USDC: {snap['balance']:.4f}\n"
                    f"  • SOL: {sol:.6f} (gas)\n"
                    f"  • Equity: {snap['equity']:.2f} USDC\n"
                    f"  • Posiciones abiertas: {snap['positions']}\n"
                    f"  • PnL total: {snap['stats']['total_pnl']:.4f} USDC",
                    parse_mode=ParseMode.HTML,
                )
                return
        await update.message.reply_text(
            f"💰 <b>Balance</b>\n"
            f"  • Efectivo: {snap['balance']:.2f} USDT\n"
            f"  • Equity: {snap['equity']:.2f} USDT\n"
            f"  • Posiciones abiertas: {snap['positions']}\n"
            f"  • PnL total: {snap['stats']['total_pnl']:.4f} USDT",
            parse_mode=ParseMode.HTML,
        )

    @authorized_only
    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        positions = self.broker.open_positions()
        if not positions:
            await update.message.reply_text("📭 Sin posiciones abiertas.")
            return
        lines = ["📊 <b>Posiciones abiertas</b>"]
        for p in positions:
            lines.append(
                f"  • {p['pair']} {p['side'].upper()} x{p['amount']:.4f} "
                f"@ {p['open_price']:.2f} (stake {p['stake']:.2f})"
            )
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

    @authorized_only
    async def cmd_pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        stats = self.db.trade_pnl_stats(self.broker.wallet_id)
        await update.message.reply_text(
            f"📈 <b>Rendimiento</b>\n"
            f"  • Trades cerrados: {stats['trades']}\n"
            f"  • Win rate: {stats['win_rate'] * 100:.1f}%\n"
            f"  • Beneficio bruto: {stats['profit']:.4f}\n"
            f"  • Pérdida bruta: {stats['loss']:.4f}\n"
            f"  • PnL neto: {stats['total_pnl']:.4f} USDT",
            parse_mode=ParseMode.HTML,
        )

    @authorized_only
    async def cmd_trade(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        args = context.args
        if len(args) != 3:
            await update.message.reply_text("Uso: /trade BTC/USDT buy 20")
            return
        pair, side, amount_str = args[0], args[1].lower(), args[2]
        try:
            amount = float(amount_str)
            if side == "buy":
                price = self.broker.price(pair)
                stake = amount
                if stake > self.broker.get_balance():
                    raise RuntimeError(f"Saldo insuficiente (tienes {self.broker.get_balance():.2f})")
                sl = price * (1 - self.s.risk.stop_loss_pct)
                tp = price * (1 + self.s.risk.take_profit_pct)
                self.broker.open_trade(pair, "long", stake, price, sl, tp)
                await update.message.reply_text(
                    f"✅ BUY {pair}: {stake:.2f} USDT @ {price:.2f}\n"
                    f"  SL {sl:.2f} / TP {tp:.2f}"
                )
            elif side == "sell":
                for pos in self.broker.positions.values():
                    if pos.pair == pair:
                        price = self.broker.price(pair)
                        result = self.broker.close_trade(pos.trade_id, price, "manual")
                        await update.message.reply_text(
                            f"✅ SELL {pair} @ {price:.2f}\n  PnL: {result.get('pnl_abs', 0):.4f} USDT"
                        )
                        return
                await update.message.reply_text(f"❌ Sin posición en {pair}")
            else:
                await update.message.reply_text("Side debe ser 'buy' o 'sell'.")
        except (ValueError, RuntimeError) as e:
            await update.message.reply_text(f"❌ {e}")

    @authorized_only
    async def cmd_agent_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self.agent.available:
            await update.message.reply_text(
                "⚠️ El agente LLM no está disponible (sin LLM_API_KEY).\n"
                "Usará la estrategia determinista (RSI) automáticamente."
            )
        if self._agent_running:
            await update.message.reply_text("🔄 El agente ya está en marcha.")
            return
        self.start_agent()
        await update.message.reply_text("🚀 Agente arrancado.")

    @authorized_only
    async def cmd_agent_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.stop_agent()
        await update.message.reply_text("🛑 Agente detenido.")

    @authorized_only
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        mode = {"paper": "🧪 Paper", "live": "💰 Live", "backtest": "📊 Backtest"}.get(self.s.mode, self.s.mode)
        agent_state = "LLM activo" if self.agent.available else "determinista (sin LLM)"
        await update.message.reply_text(
            f"🖥 <b>Estado</b>\n"
            f"  • Modo: {mode}\n"
            f"  • Agente: {agent_state}\n"
            f"  • LLM: {self.s.llm_model if self.agent.available else 'n/a'}\n"
            f"  • Loop: {'en marcha' if self._agent_running else 'parado'}\n"
            f"  • Pares: {', '.join(self.s.trading_pairs)}\n"
            f"  • Timeframe: {self.s.timeframe}",
            parse_mode=ParseMode.HTML,
        )

    @authorized_only
    async def cmd_wallet_onchain(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Muestra la cartera on-chain configurada (dirección + saldos reales)."""
        chain = context.args[0].lower() if context.args else self.s.chain
        if not self.s.wallet_private_key:
            await update.message.reply_text(
                "🔐 No hay cartera configurada.\n\n"
                "Para usar una cartera real:\n"
                "  • Crea una nueva: /create_wallet <b>base</b>|solana\n"
                "  • O pon tu WALLET_PRIVATE_KEY en el .env",
                parse_mode=ParseMode.HTML,
            )
            return
        try:
            wallet = wallet_factory(chain, self.s.wallet_rpc, self.s.wallet_private_key)
            info = wallet.get_info()
            if not info.address:
                await update.message.reply_text("❌ No se pudo derivar dirección.")
                return
            await update.message.reply_text(
                "🛰 <b>Cartera on-chain</b>\n" + info.summary(),
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:  # noqa: BLE001
            await update.message.reply_text(f"❌ Error: {e}")

    @authorized_only
    async def cmd_create_wallet(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Crea una cartera nueva en la cadena indicada y muestra la clave privada."""
        if not context.args:
            await update.message.reply_text("Uso: /create_wallet <b>base</b>|solana")
            return
        chain = context.args[0].lower()
        try:
            wallet = create_wallet(chain, self.s.wallet_rpc)
            info = wallet.get_info()
            if not info.address or not info.private_key:
                await update.message.reply_text("❌ No se pudo crear la cartera.")
                return
            await update.message.reply_text(
                "🆕 <b>Cartera creada</b>\n"
                + info.summary()
                + "\n\n🔑 <b>Clave privada:</b>\n"
                f"<code>{info.private_key}</code>\n\n"
                "⚠️ <b>GUÁRDALA EN UN LUGAR SEGURO.</b>\n"
                "Quien tenga esta clave controla la cartera. "
                "Para activarla, copia la clave en WALLET_PRIVATE_KEY (archivo .env) "
                "y usa: /wallet_onchain",
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:  # noqa: BLE001
            await update.message.reply_text(f"❌ Error: {e}")

    @authorized_only
    async def cmd_convert(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        from ..execution.live import LiveBroker
        if not isinstance(self.broker, LiveBroker):
            await update.message.reply_text("❌ /convert solo disponible en modo live.")
            return
        usdc = self.broker._auto_fund()
        if usdc > 0:
            await update.message.reply_text(
                f"✅ Convertido SOL → {usdc:.2f} USDC. Usa /wallet para ver saldo."
            )
        else:
            await update.message.reply_text("ℹ️ No hay SOL sobrante para convertir.")


def run_telegram(settings: Settings, db: Database, broker: PaperBroker, agent: LLMAgent) -> Application:
    bot = TokenScanBot(settings, db, broker, agent)

    async def _autostart(app: Application) -> None:
        bot.start_agent()

    app = Application.builder().token(settings.telegram_bot_token).post_init(_autostart).build()
    handlers = [
        ("start", bot.cmd_start),
        ("wallet", bot.cmd_wallet),
        ("deposit", bot.cmd_deposit),
        ("withdraw", bot.cmd_withdraw),
        ("balance", bot.cmd_balance),
        ("positions", bot.cmd_positions),
        ("pnl", bot.cmd_pnl),
        ("trade", bot.cmd_trade),
        ("agent_start", bot.cmd_agent_start),
        ("agent_stop", bot.cmd_agent_stop),
        ("wallet_onchain", bot.cmd_wallet_onchain),
        ("create_wallet", bot.cmd_create_wallet),
        ("convert", bot.cmd_convert),
        ("status", bot.cmd_status),
    ]
    for name, handler in handlers:
        app.add_handler(CommandHandler(name, handler))

    return app