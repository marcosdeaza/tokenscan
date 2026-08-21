# Fórmulas del sistema cuantitativo

## Indicadores técnicos

### RSI (Relative Strength Index)
```
RSI = 100 - 100 / (1 + RS)
RS = EMA(ganancias, alpha=1/14) / EMA(pérdidas, alpha=1/14)
```
Fuente: Wilder (1978). Suavizado exponencial con alpha=1/period en lugar de SMA.

### ATR (Average True Range)
```
TR = max(high - low, |high - prev_close|, |low - prev_close|)
ATR = EMA(TR, 14)
```
Fuente: Wilder (1978).

### ADX (Average Directional Index)
```
+DM = max(high - prev_high, 0) si > prev_low - low, sino 0
-DM = max(prev_low - low, 0) si > high - prev_high, sino 0
+DI = 100 * Wilder(+DM) / ATR
-DI = 100 * Wilder(-DM) / ATR
DX = 100 * |+DI - -DI| / (+DI + -DI)
ADX = Wilder(DX, 14)
```
Interpretación: ADX < 20 → rango; 20-25 → transición; ≥ 25 → tendencia.
+DI > -DI con ADX alto → tendencia alcista. -DI > +DI → bajista.

### Efficiency Ratio (Kaufman)
```
ER = |P_t - P_{t-n}| / Σ|P_i - P_{i-1}|, n = 10
```
Rango [0, 1]. ER > 0.5 → tendencia fuerte; ER < 0.3 → ruido/rango.
Fuente: Kaufman (1995), *Trading Systems and Methods*.

### MACD
```
MACD = EMA(close, 12) - EMA(close, 26)
Signal = EMA(MACD, 9)
Histogram = MACD - Signal
```

### Bollinger Bands
```
Mid = SMA(close, 20)
Std = σ(close, 20)
Upper = Mid + 2 * Std
Lower = Mid - 2 * Std
%B = (close - Lower) / (Upper - Lower)
Width = (Upper - Lower) / Mid
```

### Stochastic Oscillator
```
%K = 100 * (close - min_low_n) / (max_high_n - min_low_n), n = 14
%D = SMA(%K, 3)
```

### Volatility (anualizada)
```
σ_anual = σ(returns) * √252
```

### Rate of Change (ROC)
```
ROC = (P_t / P_{t-n}) - 1, n = 10
```

### Volume Ratio
```
VolRatio = volumen / SMA(volumen, 20)
```

### VWAP
```
VWAP = Σ(price_i * volume_i) / Σ(volume_i)
price_i = (high + low + close) / 3
```

### Price Position
```
PricePos = (close - min_n) / (max_n - min_n), n = 20
```

## Detección de régimen

Tres votos, gana por mayoría simple:

1. **ADX**: ≥ 20 → tendencia (dirección por +DI vs -DI); < 20 → rango.
2. **Efficiency Ratio**: ≥ 0.35 → tendencia (dirección por pendiente de precio); < 0.35 → rango.
3. **EMA slope**: pendiente de EMA(50) en 5 velas. Umbral plano: ±0.002.

Fuerza = votos_del_ganador / total_votos.

## Score compuesto

Cada señal emite convicción en [-1, 1]. Se ponderan y promedian sobre las que votan activamente.

### Señales individuales

- **RSI**: `(RSI - 50) / 50` → [-1, 1]
- **Stochastic %K**: `(%K - 50) / 50` → [-1, 1]
- **MACD**: `tanh(histograma / (2 * σ_hist))` → [-1, 1]
- **Bollinger %B**: `(%B - 0.5) * 2` → [-1, 1]
- **Momentum (ROC)**: `clamp(ROC / (2 * σ_ROC))` → [-1, 1]
- **EMA trend**: `clamp((EMA_fast - EMA_slow) / EMA_slow * 20)` → [-1, 1]
- **Price position**: `clamp((PricePos - 0.5) * 2)` → [-1, 1]

### Filtro ADX

Las señales de reversión (RSI, Stoch, BB) se atenúan cuando hay tendencia fuerte:
```
filter = 1.0 si ADX < 20
filter = 0.0 si ADX > 40
filter = 1 - (ADX - 20) / 20 en [20, 40]
```

### Decisión final

```
score = Σ(w_i * señal_i) / Σ(w_i)
bullish if score > 0.3
bearish if score < -0.25
hold otherwise
```

## Gestión de riesgo

### Position Sizing (ATR / Turtle)
```
risk_amount = equity * risk_per_trade_pct        # 1% del capital
notional = risk_amount / (atr_multiplier * ATR)  # 2 ATR
slot_cap = equity / (max_open_trades - open_trades)
size = min(notional, slot_cap, equity)
```

### Stop-loss y Take-profit (ATR)
```
SL_long = entry - ATR * atr_sl_multiplier   # 2 ATR
TP_long = entry + ATR * atr_tp_multiplier   # 3 ATR
SL_short = entry + ATR * atr_sl_multiplier
TP_short = entry - ATR * atr_tp_multiplier
```

### Portfolio Volatility Targeting
```
σ_real = σ(daily_returns) * √252
leverage = min(target_vol / σ_real, max_leverage)
exposure = equity * max(0, leverage)
```

### Kelly Fraction
```
b = avg_win / avg_loss
p = win_rate
f = (p * b - q) / b    # Kelly óptimo
f_used = min(f * 0.5, 0.25)   # media-Kelly, cap 25%
```

## Métricas de rendimiento

### CAGR
```
CAGR = (final / initial)^(365 / days) - 1
```

### Calmar Ratio
```
Calmar = CAGR / max_drawdown_relativo
```

### Sharpe Ratio
```
Sharpe = (mean(returns) - r_f / periods) / σ(returns) * √periods
```

### Sortino Ratio
```
Sortino = (mean(returns) - r_f / periods) / σ_downside * √periods
```

### System Quality Number (SQN) — Van Tharp
```
SQN = √(n) * mean(pnl) / σ(pnl)
```
Requiere ≥ 30 trades. Interpretación: < 1 pobre, 1.6-2.0 regular, 2.0-2.5 promedio, 2.5-3.0 bueno, 3.0-5.0 excelente, > 5 excepcional.

### Expectancy
```
E = (win_rate * avg_win) - (loss_rate * avg_loss)
```
Equivalente a la media de PnL de los trades. Positiva = edge positivo.

### Win/Loss Streaks
Racha máxima consecutiva de trades ganadores y perdedores.