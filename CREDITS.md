# CREDITS

TokenScan es un proyecto original bajo licencia MIT. Este archivo documenta las
dependencias del proyecto y la procedencia de los conceptos matemáticos que
implementa, para que cualquier persona pueda auditar y replicar el trabajo.

## Dependencias

| Paquete | Licencia | Uso |
|---------|----------|-----|
| ccxt | MIT | Acceso unificado a exchanges |
| pandas / numpy | BSD-3-Clause | Datos y cálculos numéricos |
| python-telegram-bot | LGPL-3.0 | Interfaz de Telegram |
| openai | Apache-2.0 | Cliente de modelos LLM compatibles con OpenAI (DeepSeek, OpenRouter, Groq) |
| pydantic | MIT | Configuración tipada |
| PyYAML | MIT | Archivos de configuración |

El listado completo de dependencias, con versiones, está en
[pyproject.toml](pyproject.toml).

## Matemáticas

Las fórmulas del proyecto (RSI de Wilder, EMA, ATR, MACD, Bollinger,
volatilidad anualizada, media-Kelly, drawdown, Sharpe, Sortino, profit factor)
son estándar del análisis técnico y la gestión de riesgo. Están reimplementadas
desde cero en `src/tokenscan/quant/` y `src/tokenscan/utils/pnl.py`, y cada una
lleva su referencia en [docs/formulas.md](docs/formulas.md).

## Licencia

Código original bajo [MIT](LICENSE) © 2026 Marcos de Aza. Las dependencias
conservan sus propias licencias.

> Trading e inversión en criptoactivos conllevan un riesgo elevado de pérdida
> de capital. Este proyecto es educativo y no constituye asesoramiento
> financiero. El rendimiento pasado no garantiza resultados futuros.
