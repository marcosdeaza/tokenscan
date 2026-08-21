"""Feeds de datos de mercado, noticias y on-chain."""

from __future__ import annotations

import pandas as pd

from ..execution.exchange import ExchangeClient
from ..utils.logger import setup_logger

log = setup_logger("tokenscan.data")


class MarketData:
    def __init__(self, exchange: ExchangeClient | None):
        self.exchange = exchange

    def fetch_ohlcv(self, pair: str, timeframe: str = "5m", limit: int = 500) -> pd.DataFrame:
        if self.exchange:
            raw = self.exchange.fetch_ohlcv(pair, timeframe, limit)
        else:
            raise RuntimeError("Sin conexión a exchange. Usa modo paper o configura exchange.")
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df


class NewsFeed:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def fetch_crypto_news(self, limit: int = 5) -> list[dict]:
        if not self.enabled:
            return []
        import requests
        try:
            r = requests.get(
                "https://cryptopanic.com/api/v1/posts/",
                params={"auth_token": "", "limit": limit, "filter": "important"},
                timeout=10,
            )
            if r.ok:
                return [{"title": p["title"], "url": p.get("url", "")} for p in r.json().get("results", [])]
        except Exception as e:  # noqa: BLE001
            log.warning("NewsFeed error: %s", e)
        return []


class OnChainData:
    def __init__(self, enabled: bool = False, rpc_url: str = ""):
        self.enabled = enabled
        self.rpc_url = rpc_url

    def latest_block(self) -> dict | None:
        if not self.enabled:
            return None
        try:
            from web3 import Web3
            w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            return {"block": w3.eth.block_number, "gas_price": w3.eth.gas_price}
        except ImportError:
            log.warning("web3 no instalado; pip install 'tokenscan[dex]'")
            return None
        except Exception as e:  # noqa: BLE001
            log.warning("OnChain error: %s", e)
            return None