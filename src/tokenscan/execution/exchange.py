"""Cliente de exchange real vía ccxt (usado en modo live)."""

from __future__ import annotations

import ccxt

from ..config import Settings
from ..utils.logger import setup_logger

log = setup_logger("tokenscan.exchange")


class ExchangeClient:
    def __init__(self, settings: Settings):
        self.s = settings
        ex_class = getattr(ccxt, settings.exchange_name.lower(), None)
        if not ex_class:
            raise ValueError(f"Exchange no soportado: {settings.exchange_name}")
        self.api = ex_class({
            "apiKey": settings.exchange_api_key,
            "secret": settings.exchange_api_secret,
            "enableRateLimit": True,
        })
        if settings.exchange_testnet:
            self.api.set_sandbox_mode(True)
        self.api.load_markets()
        log.info("Exchange %s conectado, %d mercados cargados",
                 settings.exchange_name, len(self.api.markets))

    def fetch_ticker(self, pair: str) -> float:
        ticker = self.api.fetch_ticker(pair)
        return ticker["last"]

    def fetch_ohlcv(self, pair: str, timeframe: str = "5m", limit: int = 500) -> list:
        """OHLCV con paginacion: ccxt/Binance limita a 1000 velas por llamada.
        Vamos hacia adelante desde el timestamp calculado hasta cubrir `limit`."""
        if limit <= 1000:
            return self.api.fetch_ohlcv(pair, timeframe, limit=limit)
        chunk = 1000
        ms = self.api.parse_timeframe(timeframe) * 1000
        start = int(self.api.milliseconds() - limit * ms)
        all_rows: list[list] = []
        since = start
        while len(all_rows) < limit:
            batch = self.api.fetch_ohlcv(pair, timeframe, limit=chunk, since=since)
            if not batch:
                break
            all_rows.extend(batch)
            if len(batch) < chunk:
                break
            since = batch[-1][0] + ms
        return all_rows[-limit:]

    def fetch_balance(self) -> dict:
        return self.api.fetch_balance()

    def create_order(self, pair: str, order_type: str, side: str, amount: float, price: float | None = None) -> dict:
        return self.api.create_order(pair, order_type, side, amount, price)

    def cancel_order(self, order_id: str, pair: str) -> dict:
        return self.api.cancel_order(order_id, pair)

    def fetch_order(self, order_id: str, pair: str) -> dict:
        return self.api.fetch_order(order_id, pair)