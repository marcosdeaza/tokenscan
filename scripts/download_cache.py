"""Descarga y cachea OHLCV historico de Binance a CSV.

Uso:
    python scripts/download_cache.py [--timeframes 1h,4h,1d] [--candles 6000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tokenscan.execution.exchange import ExchangeClient

CACHE = Path("data/cache")

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def download(pair: str, timeframe: str, candles: int) -> pd.DataFrame:
    client = ExchangeClient.__new__(ExchangeClient)
    import ccxt
    client.api = ccxt.binance({"enableRateLimit": True})
    client.api.load_markets()

    if candles <= 1000:
        raw = client.api.fetch_ohlcv(pair, timeframe, limit=candles)
    else:
        chunk = 1000
        ms = client.api.parse_timeframe(timeframe) * 1000
        start = int(client.api.milliseconds() - candles * ms)
        all_rows = []
        since = start
        while len(all_rows) < candles:
            batch = client.api.fetch_ohlcv(pair, timeframe, limit=chunk, since=since)
            if not batch:
                break
            all_rows.extend(batch)
            if len(batch) < chunk:
                break
            since = batch[-1][0] + ms
            print(f"    {pair} {timeframe}: {len(all_rows)}/{candles}")
        raw = all_rows[-candles:]

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframes", default="1h,4h,1d")
    parser.add_argument("--candles", type=int, default=6000)
    args = parser.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    for tf in args.timeframes.split(","):
        for pair in PAIRS:
            fname = f"{pair.split('/')[0]}_{tf}.csv"
            path = CACHE / fname
            if path.exists():
                print(f"[skip] {fname}")
                continue
            print(f"[fetch] {pair} {tf} ({args.candles} velas)...")
            df = download(pair, tf, args.candles)
            df.to_csv(path)
            print(f"[OK] {fname}: {len(df)} filas {df.index[0]} -> {df.index[-1]}")


if __name__ == "__main__":
    main()
