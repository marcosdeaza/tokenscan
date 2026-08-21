"""Tests de la carga de configuración (YAML anidado + env)."""

from __future__ import annotations

from tokenscan.config import Settings


def test_nested_yaml_fields_are_loaded():
    """Los dicts anidados del YAML (risk/agent/strategy/backtest) deben aplicarse."""
    data = {
        "mode": "paper",
        "timeframe": "1h",
        "interval": 60,
        "trading_pairs": ["XRP/USDT"],
        "strategy": {"name": "dummy_strategy", "rsi_oversold": 20},
        "risk": {"max_position_pct": 0.5, "risk_per_trade_pct": 0.02},
        "agent": {"chain_enabled": True, "max_trades_per_cycle": 5},
        "telegram": {"enabled": False},
        "backtest": {"days": 30},
    }
    s = Settings(**data)
    assert s.timeframe == "1h"
    assert s.interval == 60
    assert s.trading_pairs == ["XRP/USDT"]
    assert s.strategy.name == "dummy_strategy"
    assert s.strategy.rsi_oversold == 20
    assert s.risk.max_position_pct == 0.5
    assert s.risk.risk_per_trade_pct == 0.02
    assert s.agent.chain_enabled is True
    assert s.agent.max_trades_per_cycle == 5
    assert s.telegram.enabled is False
    assert s.backtest.days == 30


def test_nested_yaml_defaults_preserved():
    """Los campos no especificados conservan sus valores por defecto."""
    s = Settings(mode="paper")
    assert s.risk.max_position_pct == 0.20
    assert s.agent.chain_enabled is False
    assert s.strategy.name == "rsi_reversion"


def test_secrets_overrides_yaml():
    """Las variables de entorno ganan sobre el YAML para los secretos."""
    data = {"agent": {"chain_enabled": True}, "risk": {"max_position_pct": 0.5}}
    s = Settings(**data, chain="solana", wallet_private_key="abc")
    assert s.chain == "solana"
    assert s.wallet_private_key == "abc"
    assert s.agent.chain_enabled is True
    assert s.risk.max_position_pct == 0.5


def test_wallet_rpc_switches_by_chain():
    s = Settings(chain="solana", rpc_url="https://mainnet.base.org",
                 solana_rpc_url="https://api.mainnet-beta.solana.com")
    assert s.wallet_rpc == "https://api.mainnet-beta.solana.com"
    s2 = Settings(chain="base")
    assert s2.wallet_rpc == "https://mainnet.base.org"
