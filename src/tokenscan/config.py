"""Configuración de TokenScan: YAML + variables de entorno (.env)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]


def _load_env() -> None:
    load_dotenv(ROOT / ".env")


class RiskConfig(BaseModel):
    max_position_pct: float = 0.20
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    trailing_stop_pct: float = 0.03
    trailing_activate_pct: float = 0.04
    max_open_trades: int = 3
    max_daily_loss_pct: float = 0.10
    cooldown_minutes: int = 15


class StrategyConfig(BaseModel):
    name: str = "rsi_reversion"
    rsi_period: int = 14
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    ema_fast: int = 12
    ema_slow: int = 26


class AgentConfig(BaseModel):
    enabled: bool = True
    max_trades_per_cycle: int = 2
    risk_tolerance: float = 0.25
    news_enabled: bool = False
    chain_enabled: bool = False
    memory_entries: int = 50


class TelegramConfig(BaseModel):
    enabled: bool = True
    notify_trades: bool = True
    notify_cycles: bool = False


class BacktestConfig(BaseModel):
    days: int = 90
    initial_capital: float = 1000.0
    fee_pct: float = 0.1
    slippage_pct: float = 0.05
    output_dir: str = "results/backtest"


class Settings(BaseModel):
    mode: str = "paper"
    timeframe: str = "5m"
    interval: int = 300
    paper_capital: float = 100.0
    trading_pairs: list[str] = Field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    data_dir: str = "data"

    # Secretos (del entorno, nunca del YAML)
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    exchange_name: str = "binance"
    exchange_api_key: str = ""
    exchange_api_secret: str = ""
    exchange_testnet: bool = False
    rpc_url: str = "https://mainnet.base.org"
    wallet_private_key: str = ""

    @classmethod
    def load(cls, path: str | Path | None = None) -> Settings:
        _load_env()
        data: dict[str, Any] = {}
        cfg_path = Path(path) if path else ROOT / "config.yaml"
        if cfg_path.exists():
            data = yaml.safe_load(cfg_path.read_text()) or {}

        raw = cls._flat(data)
        secrets = {
            "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
            "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
            "llm_api_key": os.getenv("LLM_API_KEY", ""),
            "llm_base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            "llm_model": os.getenv("LLM_MODEL", "deepseek-chat"),
            "exchange_name": os.getenv("EXCHANGE_NAME", "binance"),
            "exchange_api_key": os.getenv("EXCHANGE_API_KEY", ""),
            "exchange_api_secret": os.getenv("EXCHANGE_API_SECRET", ""),
            "exchange_testnet": os.getenv("EXCHANGE_TESTNET", "false").lower() == "true",
            "rpc_url": os.getenv("RPC_URL", "https://mainnet.base.org"),
            "wallet_private_key": os.getenv("WALLET_PRIVATE_KEY", ""),
        }
        return cls(**raw, **secrets)

    @staticmethod
    def _flat(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in data.items():
            full = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                out.update(Settings._flat(value, full))
            else:
                out[full] = value
        return out

    @property
    def agent_available(self) -> bool:
        return self.agent.enabled and bool(self.llm_api_key)

    def pair_base(self, pair: str) -> str:
        return pair.split("/")[0].upper()
