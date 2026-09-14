from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize

TMP_DIR = Path(__file__).resolve().parent / "tmp"
CACHE_DIR = TMP_DIR / "cache" / "index_prices"


@dataclass(frozen=True)
class BacktestIndex:
    code: str
    name: str
    provider: str
    total_return_codes: tuple[str, ...] = ()


INDEXES = {
    "成长100": BacktestIndex("980080", "国证成长100", "国证指数", ("480080",)),
    "价值100": BacktestIndex("980081", "国证价值100", "国证指数", ("480081",)),
    "红利低波": BacktestIndex("H30269", "中证红利低波", "中证指数", ("H20269",)),
    "科创成长": BacktestIndex("000690", "中证科创成长", "中证指数", ("H30705",)),
    "创业板成长": BacktestIndex("399667", "国证创业板成长", "国证指数"),
    "自由现金流": BacktestIndex("980092", "国证自由现金流", "国证指数", ("480092",)),
}


def _number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "rows", "records", "list", "result", "items"):
            if key in payload:
                records = _records(payload[key])
                if records:
                    return records
    return []


def _parse_payload(payload: Any) -> pd.Series:
    raw_rows = None
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        raw_rows = payload["data"].get("data")
    if isinstance(raw_rows, list) and raw_rows and isinstance(raw_rows[0], list):
        if any(len(row) < 6 for row in raw_rows):
            raise ValueError("国证官网历史行情记录字段不足")
        frame = pd.DataFrame({"date": [row[0] for row in raw_rows], "price": [row[5] for row in raw_rows]})
        result = pd.DataFrame({"date": pd.to_datetime(frame["date"], errors="coerce", format="mixed"), "price": frame["price"].map(_number)})
        result = result.dropna().drop_duplicates("date").sort_values("date")
        if len(result) < 2:
            raise ValueError("官网返回的有效数据不足 2 个交易日")
        return result.set_index("date")["price"].rename("price")
    records = _records(payload)
    if not records:
        raise ValueError("官网返回中没有历史行情记录")
    if isinstance(records[0], dict) and isinstance(records[0].get("value"), list):
        rows = [record["value"] for record in records]
        if any(len(row) < 6 for row in rows):
            raise ValueError("国证官网历史行情记录字段不足")
        frame = pd.DataFrame({"date": [row[0] for row in rows], "price": [row[5] for row in rows]})
    else:
        frame = pd.DataFrame(records)
    date_col = next((c for c in frame if str(c).lower() in {"date", "tradedate", "trade_date", "日期", "交易日期"}), None)
    price_col = next((c for c in frame if str(c).lower() in {"close", "closeprice", "indexpoint", "收盘", "收盘价", "收盘点位"}), None)
    date_col = date_col or next((c for c in frame if "date" in str(c).lower() or "日期" in str(c)), None)
    price_col = price_col or next((c for c in frame if "close" in str(c).lower() or "收盘" in str(c) or "price" in str(c).lower()), None)
    if date_col is None or price_col is None:
        raise ValueError(f"无法识别日期/收盘价字段: {list(frame.columns)}")
    result = pd.DataFrame({"date": pd.to_datetime(frame[date_col], errors="coerce", format="mixed"), "price": frame[price_col].map(_number)})
    result = result.dropna().drop_duplicates("date").sort_values("date")
    if len(result) < 2:
        raise ValueError("官网返回的有效数据不足 2 个交易日")
    return result.set_index("date")["price"].rename("price")


def _cache_path(index: BacktestIndex, code: str | None = None) -> Path:
    return CACHE_DIR / f"{index.provider}_{code or index.code}.csv"


def _read_cache(index: BacktestIndex, code: str | None = None) -> pd.Series:
    path = _cache_path(index, code)
    if not path.exists():
        return pd.Series(dtype="float64", name="price")
    frame = pd.read_csv(path, parse_dates=["date"])
    if not {"date", "price"}.issubset(frame.columns):
        return pd.Series(dtype="float64", name="price")
    return frame.dropna(subset=["date", "price"]).drop_duplicates("date").set_index("date")["price"].sort_index()


def _write_cache(index: BacktestIndex, prices: pd.Series, code: str | None = None) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(index, code)
    temporary = path.with_suffix(".tmp")
    prices.rename("price").rename_axis("date").reset_index().to_csv(temporary, index=False, date_format="%Y-%m-%d")
    temporary.replace(path)


def _remote_fetch(index: BacktestIndex, code: str, start: date, end: date) -> pd.Series:
    if index.provider == "中证指数":
        candidates = [
            ("https://www.csindex.com.cn/csindex-home/perf/index-perf", {"indexCode": code, "startDate": start.isoformat(), "endDate": end.isoformat()}),
        ]
    else:
        candidates = [
            ("https://hq.cnindex.com.cn/market/market/getIndexDailyDataWithDataFormat", {"indexCode": code, "startDate": start.isoformat(), "endDate": end.isoformat(), "frequency": "day"}),
        ]
    errors = []
    for url, params in candidates:
        try:
            response = requests.get(url, params=params, timeout=30, headers={"User-Agent": "portfolio-tools-backtest/1.0"})
            response.raise_for_status()
            return _parse_payload(response.json())
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("; ".join(errors))


def load_prices(index: BacktestIndex, start: date, end: date) -> pd.Series:
    candidates = [(code, True) for code in index.total_return_codes] + [(index.code, False)]
    errors = []
    for code, is_total_return in candidates:
        try:
            result = _load_prices_variant(index, code, start, end)
            print(f"{index.name}: 使用{'全收益' if is_total_return else '价格'}指数 {code}。")
            return result
        except RuntimeError as exc:
            errors.append(str(exc))
    raise RuntimeError(f"{index.name} 所有数据口径均不可用: {'; '.join(errors)}")


def _load_prices_variant(index: BacktestIndex, code: str, start: date, end: date) -> pd.Series:
    requested_start, requested_end = pd.Timestamp(start), pd.Timestamp(end)
    cached = _read_cache(index, code)
    if not cached.empty and cached.index.min() <= requested_start and cached.index.max() >= requested_end:
        return cached.loc[requested_start:requested_end]
    fetch_start, fetch_end = start, end
    if not cached.empty:
        if cached.index.min() <= requested_start:
            fetch_start = (cached.index.max() + pd.Timedelta(days=1)).date()
        if cached.index.max() >= requested_end:
            fetch_end = (cached.index.min() - pd.Timedelta(days=1)).date()
    if fetch_start <= fetch_end:
        try:
            fresh = _remote_fetch(index, code, fetch_start, fetch_end)
        except RuntimeError:
            if cached.empty or fetch_start <= cached.index.max().date():
                raise
            print(f"{index.name} 官网暂无 {fetch_start} 至 {fetch_end} 数据，使用缓存至 {cached.index.max().date()}。")
        else:
            cached = pd.concat([cached, fresh]).sort_index()
            cached = cached[~cached.index.duplicated(keep="last")]
            _write_cache(index, cached, code)
    result = cached.loc[requested_start:requested_end]
    if result.empty:
        raise RuntimeError(f"{index.name} 没有可用数据")
    return result


def _risk_parity_weights(train_returns: pd.DataFrame) -> pd.Series:
    covariance = train_returns.cov().values * 252
    count = len(train_returns.columns)
    initial = np.ones(count) / count

    def objective(weights: np.ndarray) -> float:
        portfolio_variance = max(float(weights @ covariance @ weights), 1e-12)
        marginal_risk = covariance @ weights
        contribution = weights * marginal_risk / np.sqrt(portfolio_variance)
        target = np.full(count, contribution.sum() / count)
        return float(np.sum((contribution - target) ** 2))

    result = minimize(objective, initial, method="SLSQP", bounds=[(0, 1)] * count, constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1})
    weights = result.x if result.success else initial
    weights = np.clip(weights, 0, None)
    weights /= weights.sum()
    return pd.Series(weights, index=train_returns.columns, name="weight")


def _risk_budget_weights(train_returns: pd.DataFrame, budgets: dict[str, float] | None) -> pd.Series:
    covariance = train_returns.cov().values * 252
    count = len(train_returns.columns)
    initial = np.ones(count) / count
    if budgets is None:
        limits = np.ones(count) / count
    else:
        missing = [name for name in train_returns.columns if name not in budgets]
        if missing:
            raise ValueError(f"风险预算缺少资产: {', '.join(missing)}")
        limits = np.array([budgets[name] for name in train_returns.columns], dtype=float)
    if np.any(limits <= 0):
        raise ValueError("风险预算必须全部大于 0")

    def risk_contribution(weights: np.ndarray) -> np.ndarray:
        variance = max(float(weights @ covariance @ weights), 1e-12)
        return weights * (covariance @ weights) / variance

    constraints = [
        {"type": "eq", "fun": lambda weights: weights.sum() - 1},
        {"type": "ineq", "fun": lambda weights: limits - risk_contribution(weights)},
    ]
    result = minimize(lambda weights: float(weights @ covariance @ weights), initial, method="SLSQP", bounds=[(0, 1)] * count, constraints=constraints)
    weights = result.x if result.success else initial
    weights = np.clip(weights, 0, None)
    weights /= weights.sum()
    return pd.Series(weights, index=train_returns.columns, name="weight")


def _mpt_weights(train_returns: pd.DataFrame, objective: str, target: float) -> pd.Series:
    annual_mean = train_returns.mean().values * 252
    annual_cov = train_returns.cov().values * 252
    count = len(train_returns.columns)
    initial = np.ones(count) / count

    def volatility(weights: np.ndarray) -> float:
        return float(np.sqrt(max(weights @ annual_cov @ weights, 1e-12)))

    if objective == "max-return-at-volatility":
        constraints = [
            {"type": "eq", "fun": lambda weights: weights.sum() - 1},
            {"type": "ineq", "fun": lambda weights: target - volatility(weights)},
        ]
        target_objective = lambda weights: -float(weights @ annual_mean)
    else:
        constraints = [
            {"type": "eq", "fun": lambda weights: weights.sum() - 1},
            {"type": "ineq", "fun": lambda weights: float(weights @ annual_mean) - target},
        ]
        target_objective = lambda weights: volatility(weights)

    result = minimize(target_objective, initial, method="SLSQP", bounds=[(0, 1)] * count, constraints=constraints)
    weights = result.x if result.success else initial
    weights = np.clip(weights, 0, None)
    weights /= weights.sum()
    return pd.Series(weights, index=train_returns.columns, name="weight")


def optimize_weights(train_returns: pd.DataFrame, risk_free: float, return_weight: float, model: str = "max-sharpe") -> pd.Series:
    if model == "risk-parity":
        return _risk_parity_weights(train_returns)
    annual_mean = train_returns.mean().values * 252
    annual_cov = train_returns.cov().values * 252
    count = len(train_returns.columns)
    initial = np.ones(count) / count

    def objective(weights: np.ndarray) -> float:
        expected_return = float(weights @ annual_mean)
        volatility = float(np.sqrt(max(weights @ annual_cov @ weights, 1e-12)))
        sharpe = (expected_return - risk_free) / volatility
        return -(sharpe + return_weight * expected_return)

    result = minimize(objective, initial, method="SLSQP", bounds=[(0, 1)] * count, constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1})
    weights = result.x if result.success else initial
    weights = np.clip(weights, 0, None)
    weights /= weights.sum()
    return pd.Series(weights, index=train_returns.columns, name="weight")


def metrics(returns: pd.Series, risk_free: float) -> dict[str, float]:
    wealth = (1 + returns).cumprod()
    annual_return = float(wealth.iloc[-1] ** (252 / len(returns)) - 1)
    volatility = float(returns.std(ddof=1) * np.sqrt(252))
    sharpe = (annual_return - risk_free) / volatility if volatility else np.nan
    drawdown = wealth / wealth.cummax() - 1
    max_drawdown = float(drawdown.min())
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan
    return {"年化收益率": annual_return, "年化波动率": volatility, "夏普比率": sharpe, "最大回撤": max_drawdown, "卡玛比率": calmar}


def _parse_risk_budgets(text: str) -> dict[str, float] | None:
    if not text.strip():
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {}
        for item in text.strip().strip("{}").split(","):
            if not item.strip() or ":" not in item:
                raise ValueError("风险预算应为 JSON 对象或 key:value 列表")
            key, value = item.split(":", 1)
            parsed[key.strip().strip("'\"")]=float(value.strip())
    if not isinstance(parsed, dict):
        raise ValueError("风险预算必须是 JSON 对象")
    return {str(key): float(value) for key, value in parsed.items()}


def run_backtest(prices: pd.DataFrame, lookback: int, rebalance: int, risk_free: float, return_weight: float, model: str, mpt_objective: str, mpt_target: float, risk_budgets: dict[str, float] | None) -> tuple[pd.Series, pd.DataFrame]:
    returns = prices.pct_change().dropna()
    portfolio_returns = []
    weight_records = []
    for position in range(lookback, len(returns), rebalance):
        train = returns.iloc[position - lookback:position]
        if model == "mpt":
            weights = _mpt_weights(train, mpt_objective, mpt_target)
        elif model == "risk-budget":
            weights = _risk_budget_weights(train, risk_budgets)
        else:
            weights = optimize_weights(train, risk_free, return_weight, model)
        holding = returns.iloc[position:min(position + rebalance, len(returns))]
        portfolio_returns.append(holding @ weights)
        weight_records.append(pd.DataFrame({"date": holding.index[0], "index": weights.index, "weight": weights.values}))
    if not portfolio_returns:
        raise ValueError("有效交易日不足以完成一次回测，请缩短训练窗口或扩大日期范围")
    return pd.concat(portfolio_returns).sort_index(), pd.concat(weight_records, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="六个中证/国证指数的滚动最优组合回测")
    parser.add_argument("--indices", nargs="+", choices=INDEXES, default=list(INDEXES), help="选择一个或多个指数名称")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2015, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--lookback", type=int, default=252, help="每次优化使用的历史交易日")
    parser.add_argument("--rebalance", type=int, default=21, help="再平衡间隔交易日")
    parser.add_argument("--risk-free", type=float, default=0.02, help="年化无风险利率")
    parser.add_argument("--return-weight", type=float, default=0.0, help="收益率在综合目标中的权重，默认 0 表示纯最大夏普")
    parser.add_argument("--model", choices=["max-sharpe", "risk-parity", "risk-budget", "mpt"], default="max-sharpe", help="权重模型：最大夏普、风险平价、风险预算或均值-方差")
    parser.add_argument("--mpt-objective", choices=["max-return-at-volatility", "min-volatility-at-return"], default="max-return-at-volatility", help="MPT目标：给定波动率最大化收益，或给定收益率最小化波动率")
    parser.add_argument("--mpt-target", type=float, default=0.20, help="MPT目标值：波动率或年化收益率，小数表示，例如 0.20")
    parser.add_argument("--risk-budgets", type=str, default="", help="风险贡献上限 JSON，例如 {\"国证成长100\":0.30,\"国证价值100\":0.30,...}")
    parser.add_argument("--output-dir", type=Path, default=TMP_DIR / "backtest_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start >= args.end or args.lookback < 20 or args.rebalance < 1 or args.return_weight < 0:
        raise SystemExit("参数无效：请检查日期、lookback、rebalance 和 return-weight")
    selected = [INDEXES[name] for name in args.indices]
    series = {}
    for index in selected:
        print(f"获取/读取 {index.name} ({index.code}) ...")
        series[index.name] = load_prices(index, args.start, args.end)
    prices = pd.concat(series, axis=1).dropna()
    if args.mpt_target <= 0:
        raise SystemExit("参数无效：mpt-target 必须大于 0")
    try:
        risk_budgets = _parse_risk_budgets(args.risk_budgets)
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"risk-budgets 格式无效: {exc}") from exc
    if risk_budgets is not None:
        risk_budgets = {INDEXES.get(name, BacktestIndex("", name, "")).name: value for name, value in risk_budgets.items()}
    portfolio_returns, weights = run_backtest(prices, args.lookback, args.rebalance, args.risk_free, args.return_weight, args.model, args.mpt_objective, args.mpt_target, risk_budgets)
    result = pd.DataFrame([metrics(portfolio_returns, args.risk_free)])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(args.output_dir / "weights.csv", index=False, encoding="utf-8-sig")
    latest_date = weights["date"].max()
    latest_weights = weights[weights["date"] == latest_date].copy()
    latest_weights["weight_pct"] = latest_weights["weight"] * 100
    latest_weights["annual_return"] = result.at[0, "年化收益率"]
    latest_weights["annual_volatility"] = result.at[0, "年化波动率"]
    latest_weights["sharpe_ratio"] = result.at[0, "夏普比率"]
    latest_weights.to_csv(args.output_dir / "latest_weights.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"date": portfolio_returns.index, "portfolio_return": portfolio_returns.values, "wealth": (1 + portfolio_returns).cumprod().values}).to_csv(args.output_dir / "portfolio_curve.csv", index=False, encoding="utf-8-sig")
    print("\n回测结果")
    print(result.T.to_string(header=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\n最新组合比例（{latest_date}）")
    print(latest_weights[["index", "weight_pct"]].to_string(index=False, formatters={"weight_pct": lambda value: f"{value:.2f}%"}))
    print("\n组合指标")
    print(f"年化收益率: {result.at[0, '年化收益率']:.2%}")
    print(f"年化波动率: {result.at[0, '年化波动率']:.2%}")
    print(f"夏普比率: {result.at[0, '夏普比率']:.4f}")
    print(f"\n最优组合权重已写入: {args.output_dir / 'weights.csv'}")
    print(f"最新组合比例已写入: {args.output_dir / 'latest_weights.csv'}")
    print(f"回测净值已写入: {args.output_dir / 'portfolio_curve.csv'}")


if __name__ == "__main__":
    main()
