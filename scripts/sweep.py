"""Motor de barrido masivo de estrategias sobre datos cacheados.

Realista para el bot live: LONG-ONLY (spot SOL/USDC), con fees/slippage,
filtro de régimen opcional y exit por señal / SL / TP / trailing.

Uso:
    python scripts/sweep.py [--days 250] [--out results/sweep.json]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tokenscan.quant.indicators import add_indicators

CACHE = Path("data/cache")
PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

FEE = 0.001       # 0.1% por lado (taker)
SLIP = 0.0005     # 0.05% slippage simulado


@dataclass
class Trade:
    pair: str
    open_price: float
    close_price: float
    stake: float
    open_index: int
    close_index: int
    pnl_abs: float
    pnl_ratio: float
    exit_reason: str


@dataclass
class Result:
    name: str
    params: dict
    timeframe: str
    initial: float
    final: float
    total_return: float
    trades: list[Trade] = field(default_factory=list)
    equity: list[float] = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.pnl_abs > 0) / len(self.trades)

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.pnl_abs for t in self.trades if t.pnl_abs > 0)
        gross_loss = abs(sum(t.pnl_abs for t in self.trades if t.pnl_abs < 0))
        if gross_loss == 0:
            return float("inf") if gross_win > 0 else 0.0
        return gross_win / gross_loss

    @property
    def max_drawdown(self) -> float:
        peak = self.equity[0] if self.equity else self.initial
        mdd = 0.0
        for v in self.equity:
            peak = max(peak, v)
            mdd = max(mdd, (peak - v) / peak if peak else 0.0)
        return mdd

    @property
    def sharpe(self) -> float:
        if len(self.equity) < 3:
            return 0.0
        r = np.diff(self.equity) / np.array(self.equity[:-1], dtype=float)
        r = r[np.isfinite(r)]
        if r.size < 2 or r.std() == 0:
            return 0.0
        return float(np.mean(r) / np.std(r) * math.sqrt(len(r)))


def load_cached(pair: str, timeframe: str) -> pd.DataFrame | None:
    fname = f"{pair.split('/')[0]}_{timeframe}.csv"
    path = CACHE / fname
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")


def precompute_regime(close: pd.Series, ema_period: int = 50, lookback: int = 5,
                      eps: float = 0.003) -> np.ndarray:
    """Vectorizado: régimen por vela ('up'/'down'/'ranging') sin lookahead."""
    ema = close.ewm(span=ema_period, adjust=False).mean()
    prev = ema.shift(lookback)
    slope = (ema - prev) / prev.replace(0, np.nan)
    slope = slope.fillna(0.0).values
    n = len(slope)
    out = np.empty(n, dtype=object)
    out[: ema_period + lookback] = "ranging"
    for i in range(ema_period + lookback, n):
        s = slope[i]
        if s > eps:
            out[i] = "up"
        elif s < -eps:
            out[i] = "down"
        else:
            out[i] = "ranging"
    return out


def make_signals(df: pd.DataFrame, params: dict, reg: str | None,
                 regime: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (entry, exit) arrays booleanos para long-only, sin lookahead."""
    n = len(df)
    entry = np.zeros(n, dtype=bool)
    exit_arr = np.zeros(n, dtype=bool)
    fam = params["family"]
    close_s = df["close"]
    close = close_s.values

    mask = np.ones(n, dtype=bool)
    if reg == "not_down":
        mask = (regime != "down")
    elif reg is not None:
        mask = (regime == reg)

    # Market filter macro: solo operar si precio > EMA larga (tendencia viva).
    # Si macro_tf esta definido ('1d', '4h'), el filtro se computa sobre el close
    # de ESE timeframe superior (verdadero filtro macro de mercado) y se alinea
    # por indice de fecha al timeframe de operacion. macro_ema = periodo de la EMA.
    macro_ema = params.get("macro_ema", 0)
    macro_tf = params.get("macro_tf")
    macro_bull = None
    if macro_ema:
        if macro_tf and macro_tf != params.get("timeframe"):
            macro_close = params.get("_macro_close")
            if macro_close is not None:
                ema_macro_series = macro_close.ewm(span=macro_ema, adjust=False).mean()
                # Sin look-ahead: EMA diaria conocida solo al cerrar el día -> shift(1).
                ema_macro_series = ema_macro_series.shift(1)
                aligned = ema_macro_series.reindex(close_s.index, method="ffill")
                macro_bull = (close > aligned.values)
        else:
            ema_macro_series = close_s.ewm(span=macro_ema, adjust=False).mean()
            macro_bull = (close > ema_macro_series.values)

    if fam in ("ema_trend", "ema_ml"):
        ef = close_s.ewm(span=params["ema_fast"], adjust=False).mean().values
        es = close_s.ewm(span=params["ema_slow"], adjust=False).mean().values
        prev_above = ef[:-1] > es[:-1]
        for i in range(params["ema_slow"], n):
            if not mask[i]:
                continue
            if ef[i] > es[i] and not prev_above[i - 1]:
                entry[i] = True
            if fam == "ema_trend":
                if ef[i] < es[i] and prev_above[i - 1]:
                    exit_arr[i] = True
            else:
                if ef[i] < es[i]:
                    exit_arr[i] = True
    elif fam == "macd":
        line = close_s.ewm(span=params["macd_fast"], adjust=False).mean() - close_s.ewm(span=params["macd_slow"], adjust=False).mean()
        sig = line.ewm(span=params["macd_sig"], adjust=False).mean()
        lv, sv = line.values, sig.values
        for i in range(params["macd_slow"] + params["macd_sig"], n):
            if not mask[i]:
                continue
            if lv[i] > sv[i] and lv[i - 1] <= sv[i - 1]:
                entry[i] = True
            if lv[i] < sv[i] and lv[i - 1] >= sv[i - 1]:
                exit_arr[i] = True
    elif fam == "rsi_reversion":
        rv = df["rsi"].values
        for i in range(params["rsi_period"], n):
            if not mask[i]:
                continue
            if rv[i] < params["oversold"]:
                entry[i] = True
            if rv[i] > 50:
                exit_arr[i] = True
    elif fam == "donchian":
        high = df["high"].values
        low = df["low"].values
        d = params["donchian"]
        for i in range(d, n):
            if not mask[i]:
                continue
            if close[i] > np.max(high[i - d : i]):
                entry[i] = True
            if close[i] < np.min(low[i - d : i]):
                exit_arr[i] = True
    elif fam == "momentum":
        roc = close_s.pct_change(params["roc_period"]).values
        for i in range(params["roc_period"], n):
            if not mask[i]:
                continue
            if roc[i] > params["roc_threshold"] and roc[i - 1] <= params["roc_threshold"]:
                entry[i] = True
            if roc[i] < 0:
                exit_arr[i] = True
    elif fam == "bollinger_long":
        bb_low = df["bb_low"].values
        bb_mid = df["bb_mid"].values
        for i in range(30, n):
            if not mask[i]:
                continue
            if close[i] < bb_low[i]:
                entry[i] = True
            if close[i] > bb_mid[i]:
                exit_arr[i] = True
    elif fam == "atr_filter":
        atr = df["atr"].values
        p = params["atr_period"]
        for i in range(p, n):
            if not mask[i]:
                continue
            if atr[i] > params["atr_threshold"] * close[i] and close[i] > close[i - 1]:
                entry[i] = True
            if close[i] < close[i - p]:
                exit_arr[i] = True
    elif fam in ("bull_pullback", "bull_breakout"):
        # Gate macro: solo operar si precio > EMA200 (mercado alcista).
        # bull_pullback: comprar pullback (RSI bajo) en mercado alcista.
        # bull_breakout: comprar ruptura de EMA fast > EMA slow en mercado alcista.
        ema200 = close_s.ewm(span=params.get("ema_macro", 200), adjust=False).mean().values
        rv = df["rsi"].values if "rsi" in df.columns else add_indicators(df)["rsi"].values
        if fam == "bull_pullback":
            ef = close_s.ewm(span=params.get("ema_fast", 20), adjust=False).mean().values
            es = close_s.ewm(span=params.get("ema_slow", 50), adjust=False).mean().values
        else:
            ef = close_s.ewm(span=params.get("ema_fast", 20), adjust=False).mean().values
            es = close_s.ewm(span=params.get("ema_slow", 50), adjust=False).mean().values
        warm = max(params.get("ema_macro", 200), 50, params.get("rsi_period", 14))
        for i in range(warm, n):
            if not mask[i]:
                continue
            if fam == "bull_pullback":
                if close[i] > ema200[i] and rv[i] < params.get("oversold", 35):
                    entry[i] = True
                if close[i] < ema200[i] or rv[i] > params.get("overbought", 60):
                    exit_arr[i] = True
            else:
                if close[i] > ema200[i] and ef[i] > es[i] and ef[i - 1] <= es[i - 1]:
                    entry[i] = True
                if ef[i] < es[i]:
                    exit_arr[i] = True

    if macro_bull is not None:
        entry &= macro_bull
        exit_arr |= ~macro_bull
    return entry, exit_arr


def run_config(cfg: dict, data: dict[str, pd.DataFrame], days: int,
               macro_data: dict[str, pd.DataFrame] | None = None) -> Result:
    tf = cfg["timeframe"]
    initial = cfg.get("initial", 50.0)
    equity = initial
    trades: list[Trade] = []
    reg = cfg.get("regime")
    sl_mult = cfg.get("sl_mult", 2.0)
    tp_mult = cfg.get("tp_mult", 3.0)
    trailing_pct = cfg.get("trailing", 0.0)
    max_pos_pct = cfg.get("max_pos_pct", 0.9)
    family = cfg["family"]
    cfg.setdefault("rsi_period", 14)
    cfg.setdefault("atr_period", 14)
    cfg.setdefault("macd_fast", 12)
    cfg.setdefault("macd_slow", 26)
    cfg.setdefault("macd_sig", 9)

    end = max(df.index[-1] for df in data.values())
    start = end - pd.Timedelta(days=days)

    frames = {}
    for pair, df in data.items():
        d = df[(df.index >= start) & (df.index <= end)].copy()
        if len(d) < 60:
            continue
        d = add_indicators(d, rsi_period=cfg.get("rsi_period", 14))
        regime = precompute_regime(d["close"])
        cc = dict(cfg)
        if macro_data and pair in macro_data:
            # Para un filtro macro REAL necesitamos la EMA calculada con toda la
            # historia previa al rango de backtest (no solo dentro del rango).
            md = macro_data[pair]
            md = md[md.index <= end]
            if len(md) > 20:
                cc["_macro_close"] = md["close"]
        entry, exit_arr = make_signals(d, cc, reg, regime)
        frames[pair] = {"df": d, "entry": entry, "exit": exit_arr,
                        "atr": d["atr"].values}

    max_len = max((len(f["df"]) for f in frames.values()), default=0)
    curve = [initial]
    active: dict[str, dict] = {}

    for i in range(1, max_len):
        for pair, f in frames.items():
            df = f["df"]
            if i >= len(df):
                continue
            row = df.iloc[i]
            price = float(row["close"])
            low, high = float(row["low"]), float(row["high"])
            atr_val = float(f["atr"][i])

            pos = active.get(pair)
            if pos:
                sl, tp = pos["sl"], pos["tp"]
                closed = False
                fill = 0.0
                reason = ""
                if low <= sl:
                    fill, reason, closed = sl, "stop_loss", True
                elif high >= tp:
                    fill, reason, closed = tp, "take_profit", True
                else:
                    if trailing_pct > 0 and price > pos["peak"]:
                        pos["peak"] = price
                        pos["sl"] = max(sl, price * (1 - trailing_pct))
                        sl = pos["sl"]
                    if low <= pos["sl"]:
                        fill, reason, closed = pos["sl"], "trailing", True
                    elif f["exit"][i]:
                        fill, reason, closed = price, "signal", True
                if closed:
                    stake = pos["stake"]
                    sell_val = pos["qty"] * fill * (1 - FEE - SLIP)
                    pnl = sell_val - stake
                    trades.append(Trade(pair=pair, open_price=pos["open_price"], close_price=fill,
                                        stake=stake, open_index=pos["open_index"], close_index=i,
                                        pnl_abs=pnl, pnl_ratio=pnl / stake, exit_reason=reason))
                    equity += pnl
                    del active[pair]

            if pair not in active and f["entry"][i]:
                stake = equity * max_pos_pct
                if stake >= 1.0:
                    buy_val = stake * (1 + FEE + SLIP)
                    qty = buy_val / price
                    sl_price = price - atr_val * sl_mult if atr_val > 0 else price * 0.95
                    tp_price = price + atr_val * tp_mult if atr_val > 0 else price * 1.10
                    active[pair] = {
                        "open_price": price, "qty": qty, "stake": stake,
                        "sl": sl_price, "tp": tp_price, "peak": price,
                        "open_index": i,
                    }

        mark = equity
        for pair, pos in active.items():
            f = frames[pair]
            if i < len(f["df"]):
                px = float(f["df"].iloc[i]["close"])
                mark += pos["qty"] * px * (1 - FEE) - pos["stake"]
        curve.append(mark)

    for pair, pos in active.items():
        f = frames[pair]
        price = float(f["df"].iloc[-1]["close"])
        sell_val = pos["qty"] * price * (1 - FEE - SLIP)
        pnl = sell_val - pos["stake"]
        trades.append(Trade(pair=pair, open_price=pos["open_price"], close_price=price,
                            stake=pos["stake"], open_index=pos["open_index"],
                            close_index=len(f["df"]) - 1, pnl_abs=pnl,
                            pnl_ratio=pnl / pos["stake"], exit_reason="end"))
    if active:
        equity = curve[-1]

    return Result(name=family, params=cfg, timeframe=tf,
                  initial=initial, final=equity,
                  total_return=(equity - initial) / initial,
                  trades=trades, equity=curve)


def build_grid() -> list[dict]:
    tfs = ["1h", "4h"]
    base = [
        {"family": "ema_ml", "ema_fast": f, "ema_slow": s, "sl_mult": 2.5, "tp_mult": 4.0, "trailing": 0.02, "max_pos_pct": 0.9}
        for f, s in [(5, 20), (10, 30), (12, 26), (20, 50), (10, 40), (20, 60), (30, 70), (50, 100)]
    ]
    base += [
        {"family": "ema_trend", "ema_fast": f, "ema_slow": s, "sl_mult": 2.5, "tp_mult": 4.0, "trailing": 0.02, "max_pos_pct": 0.9}
        for f, s in [(5, 20), (10, 30), (12, 26), (20, 50), (10, 40), (20, 60)]
    ]
    base += [
        {"family": "macd", "macd_fast": 12, "macd_slow": 26, "macd_sig": s, "sl_mult": 2.5, "tp_mult": 4.0, "trailing": 0.02, "max_pos_pct": 0.9}
        for s in [5, 9, 12]
    ]
    base += [
        {"family": "rsi_reversion", "rsi_period": 14, "oversold": o, "sl_mult": 2.5, "tp_mult": 4.0, "trailing": 0.0, "max_pos_pct": 0.9}
        for o in [25, 30, 35, 40]
    ]
    base += [
        {"family": "donchian", "donchian": d, "sl_mult": 2.5, "tp_mult": 4.0, "trailing": 0.02, "max_pos_pct": 0.9}
        for d in [10, 20, 30, 40, 55]
    ]
    base += [
        {"family": "momentum", "roc_period": r, "roc_threshold": t, "sl_mult": 2.5, "tp_mult": 4.0, "trailing": 0.02, "max_pos_pct": 0.9}
        for r, t in [(6, 0.01), (12, 0.02), (20, 0.03)]
    ]
    base += [
        {"family": "bollinger_long", "sl_mult": 2.5, "tp_mult": 4.0, "trailing": 0.0, "max_pos_pct": 0.9}
    ]
    base += [
        {"family": "atr_filter", "atr_period": 14, "atr_threshold": t, "sl_mult": 2.5, "tp_mult": 4.0, "trailing": 0.02, "max_pos_pct": 0.9}
        for t in [0.012, 0.015, 0.02]
    ]
    base += [
        {"family": "bull_pullback", "ema_macro": 200, "ema_fast": 20, "ema_slow": 50,
         "oversold": o, "overbought": ob, "sl_mult": 2.5, "tp_mult": 4.0, "trailing": 0.02, "max_pos_pct": 0.9}
        for o, ob in [(30, 60), (35, 65), (40, 70)]
    ]
    base += [
        {"family": "bull_breakout", "ema_macro": 200, "ema_fast": f, "ema_slow": s,
         "sl_mult": 2.5, "tp_mult": 4.0, "trailing": 0.02, "max_pos_pct": 0.9}
        for f, s in [(20, 50), (30, 70), (50, 100)]
    ]

    grid = []
    for tf in tfs:
        for cfg in base:
            for reg in (None, "up", "not_down"):
                for macro in (0, 100, 200):
                    cc = dict(cfg)
                    cc["timeframe"] = tf
                    cc["regime"] = reg
                    cc["macro_ema"] = macro
                    if macro and tf != "1d":
                        cc["macro_tf"] = "1d"
                    grid.append(cc)
    # familias con gate macro en 1d (historial largo con bull markets)
    for cfg in base:
        if cfg["family"] not in ("bull_pullback", "bull_breakout"):
            continue
        for reg in (None, "up", "not_down"):
            for macro in (0, 50, 100, 200):
                cc = dict(cfg)
                cc["timeframe"] = "1d"
                cc["regime"] = reg
                cc["macro_ema"] = macro
                grid.append(cc)
    return grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=250)
    parser.add_argument("--out", default="results/sweep.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    data = {"1h": {}, "4h": {}, "1d": {}}
    for pair in PAIRS:
        for tf in ("1h", "4h", "1d"):
            df = load_cached(pair, tf)
            if df is not None:
                data[tf][pair] = df

    grid = build_grid()
    if args.limit:
        grid = grid[: args.limit]
    print(f"[SWEEP] {len(grid)} configs x {args.days} días")

    results = []
    t0 = time.time()
    for idx, cfg in enumerate(grid, 1):
        tf_days = 1500 if cfg["timeframe"] == "1d" else args.days
        res = run_config(cfg, data[cfg["timeframe"]], tf_days,
                         macro_data=data.get("1d"))
        results.append(res)
        if idx % 20 == 0 or idx == len(grid):
            elapsed = time.time() - t0
            print(f"  [{idx}/{len(grid)}] {elapsed:.0f}s ({elapsed/max(idx,1):.1f}s/config)")

    rows = []
    for r in results:
        rows.append({
            "family": r.name,
            "params": r.params,
            "timeframe": r.timeframe,
            "regime": r.params.get("regime"),
            "final": round(r.final, 2),
            "return_pct": round(r.total_return * 100, 2),
            "trades": r.n_trades,
            "win_rate": round(r.win_rate * 100, 1),
            "profit_factor": round(r.profit_factor, 2),
            "max_dd": round(r.max_drawdown * 100, 2),
            "sharpe": round(r.sharpe, 2),
        })
    rows.sort(key=lambda x: x["profit_factor"], reverse=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"[OK] {len(rows)} resultados en {args.out}")
    print("\nTOP 15:")
    for r in rows[:15]:
        print(f"  {r['family']:16s} tf={r['timeframe']} reg={r['regime']} "
              f"ret={r['return_pct']:+7.2f}% pf={r['profit_factor']:.2f} "
              f"trades={r['trades']:3d} wr={r['win_rate']:5.1f}% dd={r['max_dd']:6.2f}%")


if __name__ == "__main__":
    main()
