"""Deflated Sharpe Ratio (Bailey & López de Prado): ¿el resultado es real o suerte?

Un resultado es "deflated" (desinflado) cuando su Sharpe se ajusta por:
- El número de intentos (configs probadas en el sweep).
- La varianza del Sharpe estimada (asimetría y curtosis de los retornos).
- El horizonte de muestra (menos datos = más probable que sea casual).

Uso:
    python scripts/dsr.py --sharpe 2.8 --n_trials 200 --days 90
    python scripts/dsr.py --backtest-results results/sweep_90d.json --top 10

Criterio práctico (llm-quant / López de Prado): DSR >= 0.95 -> el edge es real,
no producto de probar muchas configs al azar.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _norm_cdf(x: float) -> float:
    """CDF de la normal estándar usando math.erf (sin scipy)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inversa de la CDF normal (aproximación de Acklam, error < 1e-9)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow = 0.02425
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= 1.0 - plow:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def _expected_max_sharpe(n_trials: int, variance_sharpe: float) -> float:
    """E[max SR] = sqrt(var_SR) * ((1-gamma)*Z^{-1}(1-1/N) + gamma*Z^{-1}(1-1/(N*e))).

    Approximation de Bailey & López de Prado con gamma = 0.5772 (constante de
    Euler-Mascheroni) para la distribución extremal de valores máximos.
    """
    if n_trials <= 1:
        return 0.0
    gamma = 0.5772156649
    z1 = _norm_ppf(1 - 1.0 / n_trials)
    z2 = _norm_ppf(1 - 1.0 / (n_trials * math.e))
    emc = (1 - gamma) * z1 + gamma * z2
    return math.sqrt(variance_sharpe) * emc


def dsr(sharpe: float, n_trials: int, n_observations: int, returns: list[float] | None = None) -> float:
    """Sharpe Ratio desinflado.

    P(SR > SR_0) donde SR_0 es el SR esperado del mejor de N intentos casuales.
    """
    n = max(2, n_observations)
    if returns is not None and len(returns) >= 2:
        r = np.asarray(returns, dtype=float)
        mu = r.mean()
        sigma = r.std(ddof=1) if len(r) > 1 else 0.0
        if sigma == 0:
            return 0.5
        skew = float((((r - mu) / sigma) ** 3).mean())
        kurt = float((((r - mu) / sigma) ** 4).mean()) - 3.0
    else:
        skew, kurt = 0.0, 0.0

    # Varianza del Sharpe (López de Prado, "De Flawed to Deflated SR"):
    # Var[SR] = (1 - gamma3*SR + (gamma4-1)/4 * SR^2) / (n - 1)
    gamma3 = skew
    gamma4 = kurt
    variance_sr = (1.0 - gamma3 * sharpe + (gamma4 - 1.0) / 4.0 * sharpe**2) / (n - 1)
    variance_sr = max(variance_sr, 0.0)

    sr_expected_max = _expected_max_sharpe(n_trials, variance_sr)
    if variance_sr <= 0:
        return 1.0 if sharpe > sr_expected_max else 0.5

    z = (sharpe - sr_expected_max) / math.sqrt(variance_sr)
    return float(_norm_cdf(z))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sharpe", type=float, default=None)
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--n-obs", type=int, default=None,
                        help="número de retornos (si no, 6 por día, p. ej. 4h)")
    parser.add_argument("--backtest-results", default=None,
                        help="JSON de results/sweep_*.json con 'sharpe' por config")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    if args.backtest_results:
        path = Path(args.backtest_results)
        if not path.exists():
            print(f"No existe: {path}")
            return
        rows = json.loads(path.read_text())
        total = len(rows)
        print(f"Sweep: {total} configs evaluadas")
        print(f"{'Config':<42s} {'SR':>6s}  {'DSR':>6s}  {'Veredicto':<12s}")
        for row in rows[: args.top]:
            sr = float(row.get("sharpe", 0) or 0)
            n_obs = args.n_obs or 0
            trades = int(row.get("trades", 0) or 0)
            if n_obs <= 0 and trades > 0:
                n_obs = max(2, trades)  # aproximación: 1 obs por trade
            label = f"{row.get('family','?')} {row.get('timeframe','?')} ret={row.get('return_pct',0):+.1f}%"
            d = dsr(sr, total, n_obs)
            verdict = "REAL" if d >= 0.95 else ("dudoso" if d >= 0.8 else "suerte")
            print(f"{label:<42s} {sr:6.2f}  {d:6.2f}  {verdict:<12s}")
        return

    if args.sharpe is None:
        print("Necesitas --sharpe o --backtest-results")
        return
    n_obs = args.n_obs or (6 * args.days)
    d = dsr(args.sharpe, args.n_trials, n_obs)
    verdict = "REAL" if d >= 0.95 else ("dudoso" if d >= 0.8 else "suerte")
    print(f"Sharpe={args.sharpe:.2f}  intentos={args.n_trials}  obs={n_obs}")
    print(f"DSR = {d:.3f}  ->  {verdict}")


if __name__ == "__main__":
    main()
