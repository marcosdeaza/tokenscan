# Las fórmulas de TokenScan

Este documento explica **cada fórmula** que usa TokenScan, de forma sencilla y con
la referencia de dónde la hemos tomado. Todo está implementado en `src/tokenscan/quant/`
(`indicators.py`, `risk.py`) y `src/tokenscan/utils/pnl.py`, con numpy/pandas puro
— sin TA-Lib, sin cajas negras.

---

## 1. Indicadores técnicos

### RSI — Relative Strength Index (Wilder)

Mide la velocidad y magnitud de los movimientos de precio (0–100).
**>70 sobrecomprado, <30 sobrevendido.**

```
RSI = 100 - 100 / (1 + RS)

RS   = media suavizada de ganancias / media suavizada de pérdidas
       donde la suavización usa alpha = 1/period (smoothing de Wilder)
```

- **Implementación**: `indicators.rsi()` — media con `ewm(alpha=1/period)`
- **Referencia**: J. Welles Wilder, *New Concepts in Technical Trading Systems* (1978); [Investopedia](https://www.investopedia.com/terms/r/rsi.asp)

### EMA — Exponential Moving Average

Media que da más peso a los datos recientes.

```
alpha = 2 / (period + 1)
EMA_t = Precio_t × alpha + EMA_{t-1} × (1 - alpha)
```

- **Implementación**: `indicators.ema()` — `Series.ewm(span=period)`
- **Referencia**: [Investopedia — EMA](https://www.investopedia.com/terms/e/ema.asp)

### ATR — Average True Range (Wilder)

Mide la volatilidad. Se usa para tamaños de stop-loss realistas.

```
TR   = max( High-Low, |High-PrevClose|, |Low-PrevClose| )
ATR  = EMA suavizada (alpha=1/period) de TR
```

- **Implementación**: `indicators.atr()`
- **Referencia**: Wilder (1978); [Investopedia — ATR](https://www.investopedia.com/terms/a/atr.asp)

### MACD

Detección de tendencia y momentum por cruce de EMAs.

```
MACD      = EMA(12) - EMA(26)
Señal     = EMA(9) del MACD
Histograma = MACD - Señal
```

- **Implementación**: `indicators.macd()`
- **Referencia**: Gerald Appel (1979); [Investopedia — MACD](https://www.investopedia.com/terms/m/macd.asp)

### Bollinger Bands

Banda central (SMA 20) ± 2 desviaciones estándar. El precio toca bandas →
posible reacción.

```
Mid  = SMA(close, 20)
Std  = desv. estándar poblacional (ddof=0)
Upper/Lower = Mid ± 2 × Std
```

- **Implementación**: `indicators.bollinger()`
- **Referencia**: John Bollinger (1980s); [Investopedia — Bollinger](https://www.investopedia.com/terms/b/bollingerbands.asp)

### Volatilidad anualizada

Volatilidad de los retornos diarios proyectada a un año (≈365 días).

```
Vol_anual = std(retornos, ventana 21) × √365
```

- **Implementación**: `indicators.volatility()`
- **Referencia**: estándar en finanzas cuantitativas; [Investopedia — Volatility](https://www.investopedia.com/terms/v/volatility.asp)

---

## 2. Gestión de riesgo

### Kelly Criterion

Fracción óptima del capital a arriesgar por operación.

```
f* = (p × b - q) / b

p = win rate        q = 1 - p
b = ganancia media / pérdida media
```

TokenScan usa **media-Kelly** (la mitad) con **cap del 25%** para ser conservador:
`f = min(f* × 0.5, 0.25)`.

- **Implementación**: `pnl.kelly_fraction()`, `risk.position_size()`
- **Referencia**: John L. Kelly, "A New Interpretation of Information Rate" (1956); [Investopedia](https://www.investopedia.com/terms/k/kellycriterion.asp)

### Tamaño de posición

```
stake = min(equity × f, equity / slots_libres)
       f limitado además por max_position_pct (20%)
```

- **Implementación**: `risk.position_size()`
- **Referencia**: patrón de sizing de freqtrade (`max_open_trades`, `stake_amount`)

### Stop-loss, take-profit y trailing

```
SL long  = precio_apertura × (1 - 5%)
TP long  = precio_apertura × (1 + 10%)
Trailing = mejor_precio × (1 - 3%), solo activo tras +4% de beneficio
```

- **Implementación**: `risk.stop_price()`, `risk.take_profit_price()`, `risk.trailing_stop_price()`
- **Referencia**: gestión de riesgo clásica + patrón de freqtrade (`stoploss`, `trailing_stop`)

### Halt por pérdida diaria

```
SI (PnL_diario / equity) ≤ -10%  →  el agente se detiene ese día
```

- **Implementación**: `risk.should_halt_daily_loss()`
- **Referencia**: risk management estándar (p.ej. "daily loss limit")

---

## 3. PnL y métricas

### PnL de un trade

```
PnL long  = amount × close × (1-fee) − amount × open × (1+fee)
PnL short = amount × open × (1-fee) − amount × close × (1+fee)
```

- **Implementación**: `pnl.TradeResult` (fees aplicados por los dos lados)

### Win rate

```
Win rate = trades con PnL > 0 / total de trades
```

- **Referencia**: métrica universal

### Profit factor

```
Profit factor = ganancias brutas / pérdidas brutas
```

- **Referencia**: estándar de la industria

### Max drawdown

```
DD_rel = max( (peak - valor) / peak )  sobre toda la curva de equity
```

- **Referencia**: métrica universal

### Sharpe ratio

Rentabilidad ajustada al riesgo total.

```
Sharpe = (media_retorno - risk_free) / desv_estándar_retornos × √365
```

- **Referencia**: William Sharpe (1966); [Investopedia — Sharpe](https://www.investopedia.com/terms/s/sharperatio.asp)

### Sortino ratio

Igual que Sharpe, pero solo penaliza la volatilidad **negativa** (downside).

```
Sortino = (media_retorno - risk_free) / desv_estándar_downside × √365
```

- **Referencia**: Frank Sortino; [Investopedia — Sortino](https://www.investopedia.com/terms/s/sortinoratio.asp)

---

## 4. De dónde viene todo esto

Todas las fórmulas anteriores son **de dominio público / estándar de mercado**.
Las reimplementamos desde cero en TokenScan. Los **patrones de arquitectura**
(diseño de estrategia, paper-trading, Telegram, loop de agente LLM) siguen la
filosofía de proyectos open-source que estudiamos:

| Proyecto | Qué aporta a TokenScan | Licencia |
|----------|------------------------|----------|
| [freqtrade](https://github.com/freqtrade/freqtrade) | diseño de estrategias, risk manager, paper-trading | GPL-3.0 |
| [jesse](https://github.com/jesse-ai/jesse) | motor de backtest, métricas | MIT |
| [ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) | patrón agente LLM (prompt + JSON) | MIT |
| [ccxt](https://github.com/ccxt/ccxt) | acceso unificado a exchanges | MIT |

> ⚠️ Tomamos **conceptos y matemáticas**, nunca copiamos código. Todo TokenScan
> es código original (MIT). Detalle completo en [CREDITS.md](../CREDITS.md).
