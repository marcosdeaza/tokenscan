"""Broker de ejecución real on-chain (Solana vía Jupiter).

Sustituye al broker virtual cuando mode=live: el saldo y las posiciones
son reales, y cada open/close ejecuta un swap SOL <-> USDC firmado con la
clave privada del .env. El histórico de trades sigue viviendo en SQLite
para poder reportar PnL por Telegram.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..config import Settings
from ..storage.db import Database
from ..utils.logger import setup_logger
from ..utils.notifier import TelegramNotifier
from .jupiter import LIVE_PAIRS, SOL_MINT, USDC_MINT, JupiterClient
from .paper import OpenPosition, PaperBroker

log = setup_logger("tokenscan.live")


class LiveBroker(PaperBroker):
    def __init__(self, settings: Settings, db: Database):
        # No creamos ExchangeClient ccxt: el on-chain no necesita API keys de CEX.
        PaperBroker.__init__(self, settings, db)
        if settings.mode != "live":
            raise ValueError("LiveBroker solo en mode=live")
        if settings.chain not in ("solana", "sol"):
            raise ValueError("LiveBroker on-chain requiere CHAIN=solana")
        self.jupiter = JupiterClient(settings)
        self.notifier = TelegramNotifier(settings)

    # ── Cartera real ─────────────────────────────────────────

    def ensure_wallet(self) -> int:
        label = "onchain"
        w = self.db.get_wallet_by_label(label)
        if w is None:
            try:
                self._wallet_id = self.db.create_wallet(label, "USDC", 0.0)
            except Exception:
                # Carrera con otro proceso (run + telegram arrancan juntos):
                # otro ya la creó entre el get y el insert.
                w = self.db.get_wallet_by_label(label)
                if w is None:
                    raise
                self._wallet_id = w["id"]
        else:
            self._wallet_id = w["id"]
        self.load_open_positions()
        self._auto_fund()
        return self._wallet_id

    def _auto_fund(self) -> float:
        """Convierte SOL sobrante a USDC automáticamente (deja reserva de gas).

        Si la wallet tiene SOL por encima de la reserva de gas, lo convierte
        a USDC para que el agente tenga capital operativo sin intervención.
        Devuelve el USDC obtenido (0 si no hubo nada que convertir).
        """
        keep_sol = max(self.s.jupiter.min_sol_balance * 4, 0.005)
        sol = self.jupiter.get_sol_balance()
        position_sol = sum(p.amount for p in self.positions.values())
        excess = sol - keep_sol - position_sol
        if excess <= 0.0001:
            return 0.0
        log.info("[LIVE] Auto-fund: convirtiendo %.4f SOL -> USDC (reserva gas %.4f, posiciones %.4f)",
                 excess, keep_sol, position_sol)
        try:
            result = self.jupiter.swap(SOL_MINT, USDC_MINT, excess)
        except Exception as e:  # noqa: BLE001
            log.warning("[LIVE] Auto-fund falló: %s", e)
            return 0.0
        usdc = result["out_amount"]
        self.notifier.send(
            f"🏦 <b>Auto-fund</b>\n"
            f"  • Convirtió <b>{excess:.4f} SOL</b> → <b>{usdc:.2f} USDC</b>\n"
            f"  • TX: <a href='https://solscan.io/tx/{result['signature']}'>ver en Solscan</a>"
        )
        return usdc

    def get_balance(self) -> float:
        """Efectivo disponible: USDC real on-chain."""
        return self.jupiter.get_usdc_balance()

    def get_equity(self) -> float:
        cash = self.get_balance()
        for pos in self.positions.values():
            try:
                price = self.get_price_for_pair(pos.pair)
                cash += pos.amount * price
            except Exception as e:  # noqa: BLE001
                log.warning("Equity %s: %s", pos.pair, e)
        return cash

    def get_available_capital(self) -> float:
        return self.get_balance()

    # ── Precios ──────────────────────────────────────────────

    def get_price_for_pair(self, pair: str) -> float:
        if pair in LIVE_PAIRS:
            return self.jupiter.price_sol()
        raise ValueError(f"Par no soportado on-chain: {pair}")

    def price(self, pair: str) -> float:
        if pair in LIVE_PAIRS:
            return self.jupiter.price_sol()
        raise ValueError(f"Par no soportado on-chain: {pair}")

    # ── Ejecución real ───────────────────────────────────────

    def open_trade(self, pair: str, side: str, stake: float, price: float,
                   stop_loss: float, take_profit: float, fee_pct: float = 0.001) -> OpenPosition:
        if not self.is_supported_pair(pair):
            raise RuntimeError(f"Par no soportado on-chain: {pair}")
        if side != "long":
            raise RuntimeError("LiveBroker solo soporta longs (spot)")

        stake = self._apply_safety_limits(stake, pair)
        usdc = self.get_balance()
        if usdc < stake:
            converted = self._auto_fund()
            usdc = self.get_balance()
            if usdc < stake:
                raise RuntimeError(
                    f"USDC insuficiente on-chain: {usdc:.2f} < {stake:.2f} "
                    f"(auto-fund convirtió {converted:.2f})"
                )
        sol_balance = self.jupiter.get_sol_balance()
        if sol_balance < self.s.jupiter.min_sol_balance:
            raise RuntimeError(
                f"SOL insuficiente para gas on-chain: {sol_balance:.4f} < {self.s.jupiter.min_sol_balance}"
            )

        result = self.jupiter.swap(USDC_MINT, SOL_MINT, stake)  # comprar SOL
        out_sol = result["out_amount"]
        exec_price = stake / out_sol if out_sol else price

        trade_id = self.db.open_trade(
            self.wallet_id, pair, side, exec_price, out_sol, stake,
            fee_pct, stop_loss, take_profit,
        )
        pos = OpenPosition(
            trade_id=trade_id, pair=pair, side=side, amount=out_sol,
            open_price=exec_price, stake=stake, stop_loss=stop_loss,
            take_profit=take_profit, best_price=exec_price,
            opened_at=datetime.now(timezone.utc), signature=result["signature"],
        )
        self.positions[trade_id] = pos
        self._last_prices[pair] = exec_price
        log.info("[LIVE] BUY %s %.6f SOL stake=%.2f USDC @ %.2f (tx %s)",
                 pair, out_sol, stake, exec_price, result["signature"])
        self.notifier.trade_opened(pair, side, out_sol, exec_price, stake, 0.0, "on-chain", result["signature"])
        return pos

    def close_trade(self, trade_id: int, price: float, reason: str) -> dict:
        pos = self.positions.pop(trade_id, None)
        if pos is None:
            return {}
        if not self.is_supported_pair(pos.pair):
            raise RuntimeError(f"Par no soportado on-chain: {pos.pair}")
        if pos.side != "long":
            raise RuntimeError("LiveBroker solo soporta longs")

        result = self.jupiter.swap(SOL_MINT, USDC_MINT, pos.amount)  # vender SOL
        close_price = self.jupiter.price_sol()
        ret = self.db.close_trade(trade_id, close_price, 0.001, reason)
        ret["signature"] = result["signature"]
        ret["pair"] = pos.pair
        ret["exit_reason"] = reason
        log.info("[LIVE] SELL %s #%s motivo=%s pnl=%.4f (tx %s)",
                 pos.pair, trade_id, reason, ret.get("pnl_abs", 0), result["signature"])
        self.notifier.trade_closed(pos.pair, reason, ret.get("pnl_abs", 0.0), ret.get("pnl_ratio", 0.0), result["signature"])
        return ret

    def is_supported_pair(self, pair: str) -> bool:
        return pair in LIVE_PAIRS

    def _apply_safety_limits(self, stake: float, pair: str) -> float:
        """Respeta min/max trade para proteger el capital real."""
        if stake < self.s.jupiter.min_trade_usd:
            raise RuntimeError(
                f"Operación demasiado pequeña: {stake:.2f} < min {self.s.jupiter.min_trade_usd} USDC"
            )
        if stake > self.s.jupiter.max_trade_usd:
            log.warning("[LIVE] Cap por max_trade_usd: %.2f -> %.2f",
                        stake, self.s.jupiter.max_trade_usd)
            stake = self.s.jupiter.max_trade_usd
        return stake

    # Deposit/withdraw virtuales no aplican en real
    def deposit(self, amount: float) -> float:
        raise RuntimeError("Modo live: fondéate on-chain (envía USDC a tu wallet).")

    def withdraw(self, amount: float) -> float:
        raise RuntimeError("Modo live: retira on-chain (envía USDC desde tu wallet).")
