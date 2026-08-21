# Guía educativa de TokenScan

## ¿Qué es TokenScan?

TokenScan es un **kit de agente de trading** que funciona como un bot de
Telegram. Le das dinero, y su IA (o su estrategia matemática) lo opera con el
objetivo de hacerlo crecer.

TokenScan no es solo un bot: es un **laboratorio** para que entiendas cómo
funciona la simbiosis entre inteligencia artificial y cripto.

---

## 1. El bucle del agente

Cada `interval` segundos (por defecto 5 minutos), el agente ejecuta:

1. **Recopilar datos** — precio actual, velas históricas, indicadores técnicos.
2. **Decidir** — el LLM (DeepSeek) recibe un resumen en JSON y decide: comprar,
   vender o esperar. Sin LLM, usa la estrategia RSI.
3. **Ejecutar** — crea órdenes en el broker virtual (o real si es modo live).
4. **Memorizar** — guarda la decisión, el PnL y el razonamiento en SQLite.
5. **Gestionar riesgo** — stop-loss, take-profit y trailing stop en cada ciclo.

---

## 2. Las matemáticas probadas

### Indicadores

- **RSI (Relative Strength Index)**: Wilder's smoothing. Mide si un activo está
  sobrecomprado (>70) o sobrevendido (<30). La fórmula: `RSI = 100 - 100/(1+RS)`
  donde RS es la media suavizada de ganancias / media suavizada de pérdidas.
- **EMA**: media exponencial, da más peso a los precios recientes.
  `alpha = 2/(period+1)`, luego `EMA = precio * alpha + EMA_prev * (1-alpha)`.
- **ATR (Average True Range)**: volatilidad de mercado. Útil para stop-loss.
- **MACD**: diferencia entre EMA rápida (12) y lenta (26). Señal de entrada
  cuando cruza su línea de señal (EMA 9 del MACD).
- **Bollinger Bands**: banda central (SMA) ± 2 desviaciones estándar. Precio
  tocando banda inferior → posible rebote; banda superior → posible techo.

### Riesgo

- **Kelly Criterion**: `f* = (p * b - q) / b`. TokenScan usa media-Kelly (mitad)
  con un cap del 25% del capital. Así creces sin arriesgar demasiado.
- **Stop-loss fijo**: 5% por defecto. Si el precio baja un 5%, se vende.
- **Trailing stop**: cuando el precio sube un 4%, el stop se activa y sube con
  el precio (3% por debajo del máximo alcanzado). Así dejas correr las ganancias.
- **Límite por posición**: máximo 20% del capital en una sola operación.
- **Máximo drawdown diario**: si pierdes 10% en un día, el agente se detiene.

### Métricas de backtest

- **Sharpe Ratio**: rentabilidad ajustada por riesgo. >1 es bueno, >2 es muy
  bueno. Fórmula: `(mean_return - risk_free) / std_return * sqrt(periods)`.
- **Sortino**: igual que Sharpe pero solo considera la volatilidad negativa
  (la que realmente duele).
- **Max Drawdown**: la mayor caída desde un pico hasta un valle. Mide el riesgo
  real de la estrategia.
- **Profit Factor**: ganancias brutas / pérdidas brutas. >2 es excelente.

---

## 3. Cómo funciona el agente LLM

Cuando pones una clave de API, el agente usa un LLM (DeepSeek V4 Flash por
defecto, pero cualquier API compatible con OpenAI sirve). El prompt le pasa:

- Señales técnicas (RSI, EMA, MACD, ATR, volatilidad) por cada par
- Precios actuales
- Estado de la cartera (saldo, posiciones abiertas, PnL)
- Últimas decisiones que tomó y su resultado
- Noticias de cripto (si `news_enabled: true`)
- Datos on-chain (si `chain_enabled: true`)

El LLM devuelve JSON estructurado con decisiones de compra/venta. TokenScan
las ejecuta si la confianza > 60.

---

## 4. El viaje: de paper trading a tu VPS

### Fase 1: Paper trading (hoy mismo)

```bash
cp .env.example .env && cp config.yaml.example config.yaml
# edita TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID en .env
python -m tokenscan telegram
```

Habla con tu bot en Telegram. `/deposit 100`. `/agent_start`. Observa.

### Fase 2: Backtest

```bash
python -m tokenscan backtest
```

Ajusta parámetros en `config.yaml` y repite hasta que las métricas tengan
sentido.

### Fase 3: Resultados públicos

Crea una carpeta en `results/` con capturas del bot, logs de PnL, gráficos.
Sube todo a GitHub. La transparencia construye tu marca.

### Fase 4: Dinero real (15-20 €)

Cambia `mode: live` en config.yaml. Crea API keys de Binance/Bybit/OKX (solo
trading spot, sin retiros). Pon 15-20 €. El agente opera con riesgo mínimo.
Documenta el resultado.

### Fase 5: VPS

```bash
./scripts/deploy.sh usuario@vps-ip
```

TokenScan corre 24/7 en tu VPS por menos de 5 €/mes.

---

## 5. Por qué código original

No forkiamos nada. TokenScan está escrito desde cero, inspirado en los patrones
de freqtrade, jesse y ai-hedge-fund, pero:

- **No copia código protegido** — no hereda licencias restrictivas (GPL), solo
  MIT.
- **Está diseñado para ser educativo** — cada módulo es pequeño, legible y
  autocontenido.
- **Es tuyo** — puedes modificarlo, venderlo, subirlo. Sin deudas legales.

---

## 6. Para profundizar

- [freqtrade docs](https://www.freqtrade.io) — el estándar de la industria
- [Investopedia: RSI](https://www.investopedia.com/terms/r/rsi.asp)
- [Investopedia: Kelly Criterion](https://www.investopedia.com/terms/k/kellycriterion.asp)
- [DeepSeek API docs](https://platform.deepseek.com/api-docs)