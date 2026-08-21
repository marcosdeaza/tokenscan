"""Base de datos SQLite ligera (sin dependencias externas)."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any


class Database:
    def __init__(self, path: str = "data/tokenscan.db"):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._init()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _init(self) -> None:
        with self._lock, self.conn:
            self.conn.executescript("""                CREATE TABLE IF NOT EXISTS wallets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT UNIQUE NOT NULL,
                    currency TEXT NOT NULL DEFAULT 'USDT',
                    balance REAL NOT NULL DEFAULT 0.0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    wallet_id INTEGER NOT NULL,
                    pair TEXT NOT NULL,
                    side TEXT NOT NULL CHECK(side IN ('long','short')),
                    open_price REAL NOT NULL,
                    close_price REAL,
                    amount REAL NOT NULL,
                    stake REAL NOT NULL,
                    fee_open REAL NOT NULL DEFAULT 0.001,
                    fee_close REAL DEFAULT 0.001,
                    pnl_abs REAL,
                    pnl_ratio REAL,
                    status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','closed','canceled')),
                    open_date TEXT NOT NULL,
                    close_date TEXT,
                    exit_reason TEXT,
                    stop_loss REAL DEFAULT 0.0,
                    take_profit REAL DEFAULT 0.0,
                    FOREIGN KEY (wallet_id) REFERENCES wallets(id)
                );
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id INTEGER,
                    order_id TEXT UNIQUE,
                    pair TEXT NOT NULL,
                    side TEXT NOT NULL,
                    type TEXT NOT NULL,
                    price REAL,
                    amount REAL,
                    filled REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'open',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (trade_id) REFERENCES trades(id)
                );
                CREATE TABLE IF NOT EXISTS agent_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle INTEGER NOT NULL,
                    decision TEXT NOT NULL,
                    reasoning TEXT,
                    pnl_impact REAL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS pnl_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    wallet_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    equity REAL NOT NULL,
                    daily_pnl REAL NOT NULL DEFAULT 0.0,
                    FOREIGN KEY (wallet_id) REFERENCES wallets(id)
                );
            """)
            self._migrate()

    def _migrate(self) -> None:
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(trades)").fetchall()}
        if "stop_loss" not in cols:
            self.conn.execute("ALTER TABLE trades ADD COLUMN stop_loss REAL DEFAULT 0.0")
        if "take_profit" not in cols:
            self.conn.execute("ALTER TABLE trades ADD COLUMN take_profit REAL DEFAULT 0.0")

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # --- Wallets ---
    def create_wallet(self, label: str, currency: str = "USDT", balance: float = 0.0) -> int:
        with self._lock, self.conn:
            cur = self.conn.execute(
                "INSERT INTO wallets (label, currency, balance) VALUES (?, ?, ?)",
                (label, currency, balance),
            )
            return cur.lastrowid

    def get_wallet(self, wallet_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM wallets WHERE id = ?", (wallet_id,)).fetchone()
        return dict(row) if row else None

    def get_wallet_by_label(self, label: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM wallets WHERE label = ?", (label,)).fetchone()
        return dict(row) if row else None

    def list_wallets(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM wallets").fetchall()]

    def update_balance(self, wallet_id: int, delta: float) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE wallets SET balance = balance + ? WHERE id = ?",
                (delta, wallet_id),
            )

    def set_balance(self, wallet_id: int, balance: float) -> None:
        with self._lock, self.conn:
            self.conn.execute("UPDATE wallets SET balance = ? WHERE id = ?", (balance, wallet_id))

    # --- Trades ---
    def open_trade(self, wallet_id: int, pair: str, side: str, price: float, amount: float,
                   stake: float, fee: float = 0.001, stop_loss: float = 0.0,
                   take_profit: float = 0.0) -> int:
        with self._lock, self.conn:
            cur = self.conn.execute(
                "INSERT INTO trades (wallet_id, pair, side, open_price, amount, stake, fee_open, "
                "stop_loss, take_profit, open_date, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), 'open')",
                (wallet_id, pair, side, price, amount, stake, fee, stop_loss, take_profit),
            )
            return cur.lastrowid

    def close_trade(self, trade_id: int, close_price: float, fee: float, reason: str = "signal") -> dict[str, Any]:
        with self._lock, self.conn:
            trade = self.conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
            if not trade or trade["status"] != "open":
                return {}
            t = dict(trade)
            open_val = t["amount"] * t["open_price"] * (1 + t["fee_open"] if t["side"] == "long" else 1 - t["fee_open"])
            close_val = t["amount"] * close_price * (1 - fee if t["side"] == "long" else 1 + fee)
            pnl_abs = close_val - open_val if t["side"] == "long" else open_val - close_val
            pnl_ratio = pnl_abs / open_val if open_val else 0.0
            self.conn.execute(
                "UPDATE trades SET close_price=?, fee_close=?, pnl_abs=?, pnl_ratio=?, "
                "status='closed', close_date=datetime('now'), exit_reason=? WHERE id=?",
                (close_price, fee, round(pnl_abs, 8), round(pnl_ratio, 8), reason, trade_id),
            )
            return {"trade_id": trade_id, "pnl_abs": pnl_abs, "pnl_ratio": pnl_ratio}

    def list_open_trades(self, wallet_id: int | None = None) -> list[dict[str, Any]]:
        if wallet_id:
            rows = self.conn.execute(
                "SELECT * FROM trades WHERE wallet_id=? AND status='open'", (wallet_id,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM trades WHERE status='open'").fetchall()
        return [dict(r) for r in rows]

    def list_closed_trades(self, wallet_id: int | None = None) -> list[dict[str, Any]]:
        if wallet_id:
            rows = self.conn.execute(
                "SELECT * FROM trades WHERE wallet_id=? AND status='closed' ORDER BY close_date DESC", (wallet_id,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM trades WHERE status='closed' ORDER BY close_date DESC").fetchall()
        return [dict(r) for r in rows]

    def trade_pnl_stats(self, wallet_id: int) -> dict[str, Any]:
        rows = self.conn.execute(
            "SELECT pnl_abs, pnl_ratio FROM trades WHERE wallet_id=? AND status='closed'", (wallet_id,)
        ).fetchall()
        if not rows:
            return {"trades": 0, "profit": 0, "loss": 0, "win_rate": 0, "total_pnl": 0}
        pnls = [r["pnl_abs"] for r in rows if r["pnl_abs"] is not None]
        if not pnls:
            return {"trades": 0, "profit": 0, "loss": 0, "win_rate": 0, "total_pnl": 0}
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        return {
            "trades": len(pnls),
            "profit": sum(wins),
            "loss": abs(sum(losses)),
            "win_rate": len(wins) / len(pnls) if pnls else 0,
            "total_pnl": sum(pnls),
        }

    # --- Agent memory ---
    def save_decision(self, cycle: int, decision: str, reasoning: str = "", pnl_impact: float = 0.0) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO agent_memory (cycle, decision, reasoning, pnl_impact) VALUES (?, ?, ?, ?)",
                (cycle, decision, reasoning, pnl_impact),
            )

    def get_recent_decisions(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM agent_memory ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # --- PnL log ---
    def log_pnl(self, wallet_id: int, equity: float, daily_pnl: float) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO pnl_log (wallet_id, timestamp, equity, daily_pnl) VALUES (?, datetime('now'), ?, ?)",
                (wallet_id, equity, daily_pnl),
            )

    def equity_curve(self, wallet_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT timestamp, equity FROM pnl_log WHERE wallet_id=? ORDER BY id", (wallet_id,)
        ).fetchall()
        return [dict(r) for r in rows]