"""Tests de la matemática cuantitativa: indicadores, régimen, scorer, sizing, métricas."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from tokenscan.config import RiskConfig
from tokenscan.quant.indicators import (
    add_indicators,
    adx,
    efficiency_ratio,
    price_position,
    roc,
    stochastic,
    volume_ratio,
    vwap,
)
from tokenscan.quant.regime import Regime, detect_regime
from tokenscan.quant.risk import (
    portfolio_vol_target,
    position_size,
    position_size_atr,
    stop_price_atr,
    take_profit_price_atr,
)
from tokenscan.quant.scorer import composite_score
from tokenscan.utils.pnl import (
    TradeResult,
    cagr,
    calmar_ratio,
    expectancy,
    sqn,
    win_loss_streaks,
)


def make_df(n: int = 300, start: float = 100.0, trend: float = 0.0, vol: float = 1.0) -> pd.DataFrame:
    """DataFrame sintético OHLCV con tendencia y volatilidad controladas."""
    rng = np.random.default_rng(42)
    drift = trend * np.arange(n)
    noise = rng.normal(0, vol, n)
    close = np.abs(start + drift + np.cumsum(noise))
    close = np.abs(close) + 1.0
    high = close * (1 + 0.01)
    low = close * (1 - 0.01)
    volume = rng.uniform(100, 1000, n)
    return pd.DataFrame({
        "open": close * 0.999,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def make_df_downtrend(n: int = 400, start: float = 400.0, slope: float = 0.5, vol: float = 0.4) -> pd.DataFrame:
    """Serie bajista estrictamente positiva (decrece exponencialmente)."""
    rng = np.random.default_rng(42)
    t = np.arange(n)
    close = start * np.exp(-slope * 0.01 * t) + np.cumsum(rng.normal(0, vol, n))
    close = np.abs(close) + 1.0
    return pd.DataFrame({
        "open": close * 1.001,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": rng.uniform(100, 1000, n),
    })


def make_trades(pnls: list[float]) -> list[TradeResult]:
    trades = []
    for i, pnl in enumerate(pnls):
        open_price = 100.0
        close_price = open_price + pnl
        trades.append(TradeResult(
            pair="BTC/USDT", side="long",
            open_price=open_price, close_price=close_price,
            amount=1.0, fee_open=0.0, fee_close=0.0,
            open_date=f"2024-01-{i + 1:02d}", close_date=f"2024-01-{i + 1:02d}",
            exit_reason="test",
        ))
    return trades


# ── Indicadores ──────────────────────────────────────────────


def test_add_indicators_columns():
    df = add_indicators(make_df())
    expected = {
        "rsi", "atr", "ema_fast", "ema_slow", "macd", "macd_signal", "macd_hist",
        "bb_low", "bb_mid", "bb_high", "bb_pct_b", "bb_width", "vol_ann",
        "adx", "plus_di", "minus_di", "kaufman_er", "stoch_k", "stoch_d",
        "roc", "price_pos", "vol_ratio", "vwap",
    }
    assert expected <= set(df.columns)


def test_adx_bounds():
    df = make_df()
    a, pdi, mdi = adx(df["high"], df["low"], df["close"])
    assert ((a >= 0) & (a <= 100)).all()
    assert ((pdi >= 0) & (pdi <= 100)).all()
    assert ((mdi >= 0) & (mdi <= 100)).all()


def test_adx_strong_trend_vs_flat():
    strong = adx(*_hlc(make_df(n=400, trend=2.0, vol=0.5)))[0].iloc[-1]
    flat = adx(*_hlc(make_df(n=400, trend=0.0, vol=2.0)))[0].iloc[-1]
    assert strong > flat


def _hlc(df):
    return df["high"], df["low"], df["close"]


def test_efficiency_ratio_bounds():
    er = efficiency_ratio(make_df()["close"])
    assert ((er >= 0) & (er <= 1)).all()


def test_stochastic_bounds():
    k, d = stochastic(*_hlc(make_df()))
    assert ((k >= 0) & (k <= 100)).all()
    assert ((d >= 0) & (d <= 100)).all()


def test_price_position_bounds():
    pp = price_position(make_df()["close"])
    assert ((pp >= 0) & (pp <= 1)).all()


def test_roc_vs_pct_change():
    c = make_df()["close"]
    r = roc(c, period=5)
    manual = c.pct_change(5)
    assert np.allclose(r.fillna(0), manual.fillna(0), atol=1e-6)


def test_volume_ratio_positive():
    vr = volume_ratio(make_df()["volume"])
    assert (vr.dropna() >= 0).all()


def test_vwap_between_high_low():
    df = make_df()
    v = vwap(df["high"], df["low"], df["close"], df["volume"])
    valid = v.dropna()
    assert not valid.empty


# ── Régimen ──────────────────────────────────────────────────


def test_regime_strong_up_trend():
    df = add_indicators(make_df(n=400, trend=1.5, vol=0.4))
    sig = detect_regime(df)
    assert sig.regime == Regime.TREND_UP
    assert 0 <= sig.strength <= 1


def test_regime_strong_down_trend():
    df = add_indicators(make_df_downtrend())
    sig = detect_regime(df)
    assert sig.regime == Regime.TREND_DOWN


def test_regime_ranging():
    df = add_indicators(make_df(n=400, trend=0.0, vol=2.0))
    sig = detect_regime(df)
    assert sig.regime == Regime.RANGING


def test_regime_as_dict_keys():
    df = add_indicators(make_df())
    keys = {"regime", "strength", "adx", "kaufman_er", "ema_slope"}
    assert keys <= set(detect_regime(df).as_dict().keys())


# ── Score compuesto ──────────────────────────────────────────


def test_composite_score_shape():
    df = add_indicators(make_df(n=400, trend=1.5, vol=0.4))
    out = composite_score(df)
    assert {"score", "signals", "decision", "confidence"} <= set(out.keys())
    assert -1.0 <= out["score"] <= 1.0
    assert out["decision"] in ("long", "short", "hold")
    assert 0.0 <= out["confidence"] <= 1.0


def test_composite_score_strong_bullish_signal():
    rng = np.random.default_rng(7)
    n = 300
    close = 100 + 2.0 * np.arange(n) + np.cumsum(rng.normal(0, 0.5, n))
    close = np.abs(close)
    df = pd.DataFrame({
        "open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": rng.uniform(100, 1000, n),
    })
    df = add_indicators(df)
    out = composite_score(df)
    assert out["score"] > 0


def test_composite_score_strong_bearish_signal():
    df = add_indicators(make_df_downtrend())
    out = composite_score(df)
    assert out["score"] < 0


# ── Sizing y stops ───────────────────────────────────────────


def test_position_size_atr_respects_risk():
    risk = RiskConfig()
    equity, price, atr_value = 10000.0, 100.0, 2.0
    size = position_size_atr(risk, equity, atr_value, price, risk_pct=0.01, atr_multiplier=2.0)
    expected = (equity * 0.01) / (2.0 * 2.0)
    assert math.isclose(size, expected, rel_tol=1e-6)
    assert size <= equity


def test_position_size_atr_zero_atr():
    risk = RiskConfig()
    assert position_size_atr(risk, 10000.0, 0.0, 100.0) == 0.0


def test_stop_take_price_atr():
    sl = stop_price_atr("long", 100.0, 2.0, 2.0)
    tp = take_profit_price_atr("long", 100.0, 2.0, 3.0)
    assert sl == 96.0
    assert tp == 106.0
    assert stop_price_atr("short", 100.0, 2.0, 2.0) == 104.0
    assert take_profit_price_atr("short", 100.0, 2.0, 3.0) == 94.0


def test_position_size_slot_cap():
    risk = RiskConfig(max_open_trades=2)
    size = position_size(risk, 10000.0, win_rate=0.6, avg_win=100.0, avg_loss=50.0, open_trades=1)
    assert size <= 10000.0
    assert size >= 0.0


def test_portfolio_vol_target():
    returns = [0.001] * 10
    exposure = portfolio_vol_target(10000.0, returns, target_vol=0.15, max_leverage=1.0)
    assert 0.0 <= exposure <= 10000.0


# ── Métricas ─────────────────────────────────────────────────


def test_cagr():
    assert math.isclose(cagr(100.0, 200.0, 365.0), 1.0, rel_tol=1e-6)


def test_calmar_ratio():
    assert math.isclose(calmar_ratio(0.5, 0.25), 2.0, rel_tol=1e-6)
    assert calmar_ratio(0.5, 0.0) == 0.0


def test_sqn_positive_edge():
    trades = make_trades([100, 100, 100, 100, -50, -50])
    assert sqn(trades) > 0


def test_sqn_needs_data():
    assert sqn(make_trades([10])) == 0.0


def test_expectancy():
    trades = make_trades([100, 100, -50, -50, -50])
    wins = sum(1 for t in trades if t.pnl_abs > 0)
    len(trades) - wins
    wr = wins / len(trades)
    exp = (wr * 100) - ((1 - wr) * 50)
    assert math.isclose(expectancy(trades), exp, rel_tol=1e-6)


def test_win_loss_streaks():
    streaks = win_loss_streaks(make_trades([100, 100, -50, -50, -50, 100]))
    assert streaks["max_win_streak"] == 2
    assert streaks["max_loss_streak"] == 3
