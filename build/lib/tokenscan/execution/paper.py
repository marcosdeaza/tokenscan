"""Abstracción de ejecución: paper-trading (modo real con ccxt opcional)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..config import Settings
from ..storage.db import Database
from ..utils.logger import setup_logger
from .exchange import ExchangeClient

log = setup_logger("tokenscan.execution")


@dataclass
class OpenPosition:
    trade_id: int
    pair: str
    side: str
    amount: float
    open_price: float
    stake: float
    stop_loss: float
    take_profit: float
    best_price: float
    opened_at: datetime = field(default_factory=datetime.utcnow)


class PaperBroker:
    """Broker virtual: wallet, posiciones, fills simulados con fees + slippage."""

    def __init__(self, settings: Settings, db: Database):
        self.s = settings
        self.db = db
        self.positions: dict[int, OpenPosition] = {}
        self._wallet_id: int | None = None
        self._last_prices: dict[str, float] = {}
        self.exchange: ExchangeClient | None = None
        if settings.mode == "live":
            self.exchange = ExchangeClient(settings)

    @property
    def wallet_id(self) -> int:
        if self._wallet_id is None:
            self._wallet_id = self.db.get_wallet_by_label("default")["id"]
        return self._wallet_id

    def ensure_wallet(self) -> int:
        label = "default"
        w = self.db.get_wallet_by_label(label)
        if w is None:
            self._wallet_id = self.db.create_wallet(label, "USDT", self.s.paper_capital)
        else:
            self._wallet_id = w["id"]
        return self._wallet_id

    def get_balance(self) -> float:
        w = self.db.get_wallet(self.wallet_id)
        return w["balance"] if w else 0.0

    def get_equity(self) -> float:
        cash = self.get_balance()
        invested = sum(p.stake for p in self.positions.values())
        return cash + invested

    def deposit(self, amount: float) -> float:
        if amount <= 0:
            raise ValueError("El ingreso debe ser positivo")
        self.db.update_balance(self.wallet_id, amount)
        return self.get_balance()

    def withdraw(self, amount: float) -> float:
        if amount <= 0 or amount > self.get_balance():
            raise ValueError("Retiro inválido")
        self.db.update_balance(self.wallet_id, -amount)
        return self.get_balance()

    def update_price(self, pair: str, price: float) -> None:
        self._last_prices[pair] = price

    def price(self, pair: str) -> float:
        if pair in self._last_prices:
            return self._last_prices[pair]
        if self.exchange:
            return self.exchange.fetch_ticker(pair)
        raise RuntimeError(f"Sin precio para {pair}")

    def apply_fee_and_slippage(self, price: float, side: str) -> float:
        slip = self.s.backtest.slippage_pct / 100.0
        if side == "long":
            return price * (1 + slip)
        return price * (1 - slip)

    def open_trade(self, pair: str, side: str, stake: float, price: float,
                   stop_loss: float, take_profit: float, fee_pct: float = 0.001) -> OpenPosition:
        if self.get_balance() < stake:
            raise RuntimeError(f"Saldo insuficiente: {self.get_balance():.2f} < {stake:.2f}")
        exec_price = self.apply_fee_and_slippage(price, side)
        amount = stake / exec_price
        trade_id = self.db.open_trade(self.wallet_id, pair, side, exec_price, amount, stake, fee_pct)
        self.db.update_balance(self.wallet_id, -stake)
        pos = OpenPosition(
            trade_id=trade_id, pair=pair, side=side, amount=amount,
            open_price=exec_price, stake=stake, stop_loss=stop_loss,
            take_profit=take_profit, best_price=exec_price,
        )
        self.positions[trade_id] = pos
        log.info("[BROKER] %s %s %s %.4f @ %.2f (SL %.2f / TP %.2f)",
                 side.upper(), amount, pair, stake, exec_price, stop_loss, take_profit)
        return pos

    def close_trade(self, trade_id: int, price: float, reason: str) -> dict:
        pos = self.positions.pop(trade_id, None)
        if pos is None:
            return {}
        exec_price = self.apply_fee_and_slippage(price, pos.side)
        result = self.db.close_trade(trade_id, exec_price, 0.001, reason)
        cash_back = pos.stake + result.get("pnl_abs", 0.0)
        self.db.update_balance(self.wallet_id, cash_back)
        log.info("[BROKER] CERRADO %s #%s motivo=%s pnl=%.4f (%.2f%%)",
                 pos.pair, trade_id, reason, result.get("pnl_abs", 0), result.get("pnl_ratio", 0) * 100)
        return result

    def check_exits(self, prices: dict[str, float]) -> list[dict]:
        """Evalúa stop-loss / take-profit / trailing sobre las posiciones abiertas."""
        exits: list[dict] = []
        for trade_id, pos in list(self.positions.items()):
            price = prices.get(pos.pair) or self._last_prices.get(pos.pair)
            if price is None:
                continue
            if pos.side == "long":
                pos.best_price = max(pos.best_price, price)
                if price <= pos.stop_loss:
                    exits.append(self.close_trade(trade_id, pos.stop_loss, "stop_loss"))
                    continue
                if price >= pos.take_profit:
                    exits.append(self.close_trade(trade_id, pos.take_profit, "take_profit"))
                    continue
                if pos.best_price > self.s.risk.trailing_activate_pct * pos.open_price + pos.open_price:
                    new_stop = pos.best_price * (1 - self.s.risk.trailing_stop_pct)
                    if new_stop > pos.stop_loss:
                        pos.stop_loss = new_stop
                        self._update_stop(trade_id, new_stop)
                        log.debug("[BROKER] trailing SL %s -> %.2f", pos.pair, new_stop)
            else:
                pos.best_price = min(pos.best_price, price)
                if price >= pos.stop_loss:
                    exits.append(self.close_trade(trade_id, pos.stop_loss, "stop_loss"))
                    continue
                if price <= pos.take_profit:
                    exits.append(self.close_trade(trade_id, pos.take_profit, "take_profit"))
                    continue
                if pos.best_price < pos.open_price * (1 - self.s.risk.trailing_activate_pct):
                    new_stop = pos.best_price * (1 + self.s.risk.trailing_stop_pct)
                    if new_stop < pos.stop_loss:
                        pos.stop_loss = new_stop
                        self._update_stop(trade_id, new_stop)
        return exits

    def _update_stop(self, trade_id: int, stop: float) -> None:
        # La BD guarda el SL; por simplicidad actualizamos en memoria y anotamos en la trade.
        self.db.conn.execute(
            "UPDATE trades SET exit_reason=exit_reason WHERE id=?", (trade_id,)
        )

    def open_positions(self) -> list[dict]:
        return [self.db.get_wallet and dict(t) for t in self.db.list_open_trades(self.wallet_id)]

    def to_snapshot(self) -> dict:
        return {
            "balance": self.get_balance(),
            "equity": self.get_equity(),
            "positions": len(self.positions),
            "stats": self.db.trade_pnl_stats(self.wallet_id),
        }


def create_broker(settings: Settings, db: Database) -> PaperBroker:
    return PaperBroker(settings, db)