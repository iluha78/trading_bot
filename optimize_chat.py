import pandas as pd
import numpy as np
from itertools import product
from dataclasses import dataclass
from contextlib import redirect_stdout
import io
import matplotlib.pyplot as plt

from backtest import Backtester
from demo_backtest import generate_synthetic_data

import strategy


# =========================
# Params
# =========================
@dataclass(frozen=True)
class StrategyParams:
    adx_threshold: int
    risk_reward: float
    risk_per_trade: float


# =========================
# Helpers: safe conversions
# =========================
def _to_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _series_numeric(values) -> pd.Series | None:
    """Convert values (list/series) to numeric Series safely."""
    if values is None:
        return None
    try:
        s = pd.Series(values)
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) < 2:
            return None
        return s.astype("float64")
    except Exception:
        return None


# =========================
# Equity extraction (robust)
# =========================
def extract_equity_curve(backtester, initial_capital: float, trades_df: pd.DataFrame) -> pd.Series | None:
    """
    Пытаемся достать equity curve из backtester в популярных форматах.
    Если не получается — строим из trades_df['pnl'].
    """

    # 1) Частые варианты: capital_history / equity / equity_curve / capital_curve
    candidates = [
        "capital_history",
        "capital_curve",
        "equity",
        "equity_curve",
        "equity_history",
        "balance_history",
    ]

    for name in candidates:
        if hasattr(backtester, name):
            raw = getattr(backtester, name)
            if raw is None:
                continue

            # a) raw — список чисел
            s = _series_numeric(raw)
            if s is not None:
                return s

            # b) raw — список dict'ов: [{'capital':..}, ...] / [{'equity':..}, ...]
            if isinstance(raw, (list, tuple)) and len(raw) > 1 and isinstance(raw[0], dict):
                for key in ("capital", "equity", "balance", "value"):
                    extracted = [d.get(key) for d in raw if isinstance(d, dict) and d.get(key) is not None]
                    s2 = _series_numeric(extracted)
                    if s2 is not None:
                        return s2

    # 2) Если backtester хранит trades как list[dict], иногда там есть 'capital_after'
    if trades_df is not None and not trades_df.empty:
        for key in ("capital_after", "equity_after", "balance_after"):
            if key in trades_df.columns:
                s = _series_numeric(trades_df[key].values)
                if s is not None:
                    return s

    # 3) Fallback: строим equity из pnl
    return build_equity_from_pnl(initial_capital, trades_df)


def build_equity_from_pnl(initial_capital: float, trades_df: pd.DataFrame) -> pd.Series | None:
    """
    Строим equity curve из trades_df['pnl'].
    pnl должен быть в абсолюте (деньги). Если pnl в %, будет кривая — но не упадёт.
    """
    if trades_df is None or trades_df.empty:
        return None
    if "pnl" not in trades_df.columns:
        return None

    pnl = pd.to_numeric(trades_df["pnl"], errors="coerce").dropna()
    if len(pnl) < 2:
        return None

    equity = initial_capital + pnl.cumsum()
    equity = pd.to_numeric(equity, errors="coerce").dropna()
    if len(equity) < 2:
        return None
    return equity.astype("float64")


# =========================
# Metrics
# =========================
def max_drawdown_percent(equity: pd.Series | None) -> float:
    if equity is None or len(equity) < 2:
        return np.nan
    equity = pd.to_numeric(equity, errors="coerce").dropna()
    if len(equity) < 2:
        return np.nan
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min() * 100.0)  # отрицательное значение


def sharpe_trade_to_trade(equity: pd.Series | None) -> float:
    """
    Sharpe по переходам между сделками (не annualized).
    Если у тебя есть дневные/минутные returns — лучше считать по ним,
    но это безопасный вариант.
    """
    if equity is None or len(equity) < 3:
        return np.nan
    equity = pd.to_numeric(equity, errors="coerce").dropna()
    if len(equity) < 3:
        return np.nan
    rets = equity.pct_change().dropna()
    std = rets.std(ddof=0)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(rets.mean() / std)


def win_rate_percent(trades_df: pd.DataFrame) -> float:
    if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
        return np.nan
    pnl = pd.to_numeric(trades_df["pnl"], errors="coerce").dropna()
    if len(pnl) == 0:
        return np.nan
    return float((pnl > 0).mean() * 100.0)


def profit_factor(trades_df: pd.DataFrame) -> float:
    if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
        return np.nan
    pnl = pd.to_numeric(trades_df["pnl"], errors="coerce").dropna()
    if len(pnl) == 0:
        return np.nan
    gp = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    if gl == 0:
        return float(np.inf) if gp > 0 else np.nan
    return float(gp / gl)


def expectancy(trades_df: pd.DataFrame) -> float:
    if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
        return np.nan
    pnl = pd.to_numeric(trades_df["pnl"], errors="coerce").dropna()
    if len(pnl) == 0:
        return np.nan
    return float(pnl.mean())


# =========================
# Walk-forward splits
# =========================
def walk_forward_splits(df: pd.DataFrame, train_bars: int, test_bars: int, step_bars: int):
    n = len(df)
    i = 0
    while True:
        train_end = i + train_bars
        test_end = train_end + test_bars
        if test_end > n:
            break
        yield df.iloc[i:train_end].copy(), df.iloc[train_end:test_end].copy()
        i += step_bars


# =========================
# Strategy builder (NO config mutation)
# =========================
def build_strategy(params: StrategyParams):
    """
    Предпочтительно, чтобы TrendStrategy умела принимать параметры.
    Если не умеет — fallback: выставляем атрибуты, если они существуют.
    """
    try:
        return strategy.TrendStrategy(
            adx_threshold=params.adx_threshold,
            take_profit_ratio=params.risk_reward,
            risk_per_trade=params.risk_per_trade,
        )
    except TypeError:
        s = strategy.TrendStrategy()
        # Попробуем типичные имена
        for name, value in (
            ("adx_threshold", params.adx_threshold),
            ("ADX_THRESHOLD", params.adx_threshold),
            ("take_profit_ratio", params.risk_reward),
            ("TAKE_PROFIT_RATIO", params.risk_reward),
            ("risk_per_trade", params.risk_per_trade),
            ("RISK_PER_TRADE", params.risk_per_trade),
        ):
            if hasattr(s, name):
                try:
                    setattr(s, name, value)
                except Exception:
                    pass
        return s


# =========================
# Backtest runner (silent)
# =========================
def run_silent_backtest(df: pd.DataFrame, params: StrategyParams, initial_capital: float = 100000.0) -> dict:
    backtester = Backtester(initial_capital=initial_capital)
    backtester.strategy = build_strategy(params)

    # подавляем вывод
    with redirect_stdout(io.StringIO()):
        backtester.run_backtest(df, ticker="WF")

    trades = getattr(backtester, "trades", None)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    final_capital = _to_float(getattr(backtester, "capital", np.nan))
    ret_pct = ((final_capital - initial_capital) / initial_capital) * 100.0 if np.isfinite(final_capital) else np.nan

    equity = extract_equity_curve(backtester, initial_capital, trades_df)

    return {
        "Return%": ret_pct,
        "FinalCapital": final_capital,
        "Trades": int(len(trades_df)),
        "WinRate%": win_rate_percent(trades_df),
        "ProfitFactor": profit_factor(trades_df),
        "Expectancy": expectancy(trades_df),
        "MaxDD%": max_drawdown_percent(equity),
        "Sharpe": sharpe_trade_to_trade(equity),
    }


# =========================
# Optimization
# =========================
def optimize_parameters():
    print("=" * 80)
    print("STRATEGY OPTIMIZATION (WALK-FORWARD, NO CONFIG MUTATION)")
    print("=" * 80)

    # Данные (синтетика). Дальше лучше заменить на реальные.
    df = generate_synthetic_data(days=365, volatility=0.015)

    # Grid
    adx_thresholds = [25, 28, 30, 32, 35]
    risk_rewards = [2.0, 2.5, 3.0, 3.5]
    risk_per_trades = [0.01, 0.02, 0.03, 0.04]

    grid = [
        StrategyParams(adx, rr, risk)
        for adx, rr, risk in product(adx_thresholds, risk_rewards, risk_per_trades)
    ]

    # Walk-forward окна
    # Подстрой под свою частоту данных:
    # Если минутки — тут нужны совсем другие числа (например train=30_000 баров).
    train_bars = int(len(df) * 0.6)
    test_bars = int(len(df) * 0.2)
    step_bars = test_bars

    splits = list(walk_forward_splits(df, train_bars, test_bars, step_bars))
    if not splits:
        raise ValueError("Not enough data for walk-forward. Increase df length or reduce window sizes.")

    print(f"\nData bars: {len(df)}")
    print(f"WF splits: {len(splits)} | train={train_bars}, test={test_bars}, step={step_bars}")
    print(f"Testing {len(grid)} combinations...\n")

    results = []
    for idx, params in enumerate(grid, start=1):
        split_rows = []
        for _, test_df in splits:
            m = run_silent_backtest(test_df, params, initial_capital=100000.0)
            split_rows.append(m)

        avg = pd.DataFrame(split_rows).mean(numeric_only=True).to_dict()

        row = {
            "ADX": params.adx_threshold,
            "RiskReward": params.risk_reward,
            "Risk%": params.risk_per_trade * 100.0,
            **avg,
        }

        # Score: доходность + качество - риск
        # (веса можешь менять)
        ret = _to_float(row.get("Return%"), 0.0)
        dd = abs(_to_float(row.get("MaxDD%"), 0.0))  # MaxDD отрицательный
        pf = _to_float(row.get("ProfitFactor"), 0.0)
        wr = _to_float(row.get("WinRate%"), 0.0)
        sh = _to_float(row.get("Sharpe"), 0.0)

        # PF ограничим, чтобы inf не взорвало рейтинг
        pf_clipped = min(pf, 5.0) if np.isfinite(pf) else 0.0

        row["Score"] = (
            ret * 0.45 +
            pf_clipped * 10.0 * 0.20 +
            wr * 0.10 +
            sh * 10.0 * 0.05 -
            dd * 0.20
        )

        results.append(row)

        if idx % 10 == 0:
            print(f"Progress: {idx}/{len(grid)} ({idx/len(grid)*100:.1f}%)")

    results_df = pd.DataFrame(results).sort_values("Score", ascending=False)

    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS (by Score):")
    print("=" * 80)

    cols = [
        "ADX", "RiskReward", "Risk%", "Score",
        "Return%", "MaxDD%", "Sharpe", "ProfitFactor", "WinRate%", "Trades"
    ]
    print(results_df[cols].head(10).to_string(index=False))

    best = results_df.iloc[0]
    print("\n" + "=" * 80)
    print("🏆 BEST CONFIGURATION:")
    print("=" * 80)
    print(f"ADX Threshold:   {int(best['ADX'])}")
    print(f"Risk/Reward:     {best['RiskReward']}")
    print(f"Risk per Trade:  {best['Risk%']:.2f}%")
    print(f"Score:           {best['Score']:.2f}")
    print(f"Avg Return:      {best['Return%']:.2f}%")
    print(f"Avg MaxDD:       {best['MaxDD%']:.2f}%")
    print(f"Avg Sharpe:      {best['Sharpe']:.3f}")
    print(f"Avg ProfitFactor:{best['ProfitFactor']:.2f}")
    print(f"Avg Win Rate:    {best['WinRate%']:.2f}%")
    print(f"Avg Trades:      {best['Trades']:.0f}")

    # =========================
    # Visualization (NO tight_layout)
    # =========================
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)

    results_df.groupby("ADX")["Return%"].mean().plot(ax=axes[0, 0], marker="o")
    axes[0, 0].set_title("Avg Return vs ADX Threshold")
    axes[0, 0].set_xlabel("ADX Threshold")
    axes[0, 0].set_ylabel("Average Return %")
    axes[0, 0].grid(True)

    results_df.groupby("RiskReward")["Return%"].mean().plot(ax=axes[0, 1], marker="o")
    axes[0, 1].set_title("Avg Return vs Risk/Reward")
    axes[0, 1].set_xlabel("Risk/Reward")
    axes[0, 1].set_ylabel("Average Return %")
    axes[0, 1].grid(True)

    results_df.groupby("Risk%")["Return%"].mean().plot(ax=axes[1, 0], marker="o")
    axes[1, 0].set_title("Avg Return vs Risk per Trade")
    axes[1, 0].set_xlabel("Risk per Trade %")
    axes[1, 0].set_ylabel("Average Return %")
    axes[1, 0].grid(True)

    sc = axes[1, 1].scatter(
        results_df["Trades"],
        results_df["WinRate%"],
        c=results_df["Score"],
        alpha=0.7
    )
    axes[1, 1].set_title("Win Rate vs Trades (color = Score)")
    axes[1, 1].set_xlabel("Number of Trades")
    axes[1, 1].set_ylabel("Win Rate %")
    axes[1, 1].grid(True)

    fig.colorbar(sc, ax=axes[1, 1], label="Score")

    plt.savefig("optimization_results.png", dpi=300)
    results_df.to_csv("optimization_results.csv", index=False)
    print("\n📊 Saved: optimization_results.png")
    print("📄 Saved: optimization_results.csv")

    return best


if __name__ == "__main__":
    optimize_parameters()
