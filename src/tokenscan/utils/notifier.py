"""Notificador a Telegram para reportar cada operación del agente en vivo."""

from __future__ import annotations

import requests

from ..config import Settings
from ..utils.logger import setup_logger

log = setup_logger("tokenscan.notifier")


class TelegramNotifier:
    def __init__(self, settings: Settings):
        self.s = settings
        self.enabled = bool(
            settings.telegram.enabled
            and settings.telegram_bot_token
            and settings.telegram_chat_id
        )

    def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{self.s.telegram_bot_token}/sendMessage",
                json={"chat_id": self.s.telegram_chat_id, "text": text, "parse_mode": "HTML"},
                timeout=15,
            )
            return bool(r.ok)
        except Exception as e:  # noqa: BLE001
            log.warning("Telegram send error: %s", e)
            return False

    def trade_opened(self, pair: str, side: str, qty: float, price: float,
                     stake: float, conf: float, reasoning: str = "", signature: str = "") -> None:
        base = pair.split("/")[0]
        emoji = "🟢 BUY" if side == "long" else "🔴 SELL"
        lines = [
            f"<b>{emoji} {pair}</b>",
            f"  • Cantidad: {qty:.4f} {base}",
            f"  • Precio: {price:.2f} USDC",
            f"  • Capital: {stake:.2f} USDC",
            f"  • Confianza: {conf:.0f}%",
        ]
        if reasoning:
            lines.append(f"  • Razón: {reasoning[:120]}")
        if signature:
            lines.append(f"  • TX: <a href='https://solscan.io/tx/{signature}'>ver en Solscan</a>")
        self.send("\n".join(lines))

    def trade_closed(self, pair: str, reason: str, pnl_abs: float, pnl_ratio: float,
                     signature: str | None = None) -> None:
        arrow = "📈" if pnl_abs >= 0 else "📉"
        lines = [
            f"{arrow} <b>CIERRE {pair}</b> — {reason}",
            f"  • PnL: {pnl_abs:+.4f} USDC ({pnl_ratio * 100:+.2f}%)",
        ]
        if signature:
            lines.append(f"  • TX: <a href='https://solscan.io/tx/{signature}'>ver en Solscan</a>")
        self.send("\n".join(lines))

    def cycle_summary(self, n_ops: int) -> None:
        self.send(f"🔄 Ciclo del agente: {n_ops} operaciones.")
