# CRÉDITOS

**TokenScan** es un proyecto original escrito desde cero, pero está inspirado en el
trabajo y las matemáticas probadas de la comunidad open-source de trading cuantitativo.
Este kit no copia código de ninguno de estos proyectos: reimplementa fórmulas y
patrones estándar de mercado de forma original, bajo licencia MIT.

## Inspiración y referencia educativa

| Proyecto | Autor | Licencia | Qué tomamos de ellos |
|----------|-------|----------|----------------------|
| [freqtrade](https://github.com/freqtrade/freqtrade) | freqtrade | GPL-3.0 | Patrón de estrategia (indicators → entry/exit), gestión de riesgo (stop-loss/trailing/ROI), paper-trading y estructura del bot de Telegram |
| [ccxt](https://github.com/ccxt/ccxt) | ccxt | MIT | Librería usada como dependencia para conectarse a los exchanges |
| [jesse](https://github.com/jesse-ai/jesse) | jesse-ai | MIT | Patrón de motor de backtest (loop de velas, fills, PnL, métricas) |
| [ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) | virattt | MIT | Patrón de agente LLM: datos → indicadores → resumen JSON → decisión final del LLM |
| [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) | PTB | LGPL-3.0 | Librería usada como dependencia para el bot de Telegram |

## Dependencias principales (con sus licencias)

- **ccxt** — MIT — acceso unificado a 100+ exchanges
- **pandas / numpy** — BSD-3-Clause — datos y matemáticas
- **python-telegram-bot** — LGPL-3.0 — interfaz de Telegram
- **openai** — Apache-2.0 — cliente del modelo LLM (compatible con DeepSeek y otros)
- **pydantic** — MIT — configuración tipada
- **PyYAML** — MIT — archivos de configuración

## Fórmulas matemáticas

Todas las fórmulas implementadas (RSI de Wilder, EMA, ATR, MACD, Bollinger,
volatilidad anualizada, media-Kelly, drawdown, Sharpe, Sortino, profit factor)
son fórmulas estándar del análisis técnico y la gestión de riesgo, de dominio
público, reimplementadas en `tokenscan/quant/` y `tokenscan/utils/pnl.py`.

## Nota sobre el dinero

Este proyecto es **educativo**. Trading con cripto puede hacerte perder todo tu
capital. Usa primero el modo paper. Cualquier operación con dinero real es bajo
tu propia responsabilidad.
