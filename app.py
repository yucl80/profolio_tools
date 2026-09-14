from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None


st.set_page_config(page_title="指数组合实验室", page_icon="◈", layout="wide")

CSI_HOME = "https://www.csindex.com.cn/"
CNI_HOME = "https://www.cnindex.com.cn/"
CACHE_DIR = Path(__file__).resolve().parent / "tmp" / "cache" / "index_prices"


@dataclass(frozen=True)
class IndexSpec:
    code: str
    name: str
    provider: str


INDEX_CATALOG = [
    IndexSpec("000300", "沪深300", "中证指数"),
    IndexSpec("000905", "中证500", "中证指数"),
    IndexSpec("000852", "中证1000", "中证指数"),
    IndexSpec("000688", "科创50", "中证指数"),
    IndexSpec("000922", "中证红利", "中证指数"),
    IndexSpec("399001", "深证成指", "国证指数"),
    IndexSpec("399006", "创业板指", "国证指数"),
    IndexSpec("399330", "深证100", "国证指数"),
    IndexSpec("399673", "创业板50", "国证指数"),
    IndexSpec("399324", "深证红利", "国证指数"),
    IndexSpec("518880", "黄金ETF华安", "东方财富ETF"),
]


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
                found = _records(payload[key])
                if found:
                    return found
    return []


def _frame_from_payload(payload: Any) -> pd.DataFrame:
    records = _records(payload)
    if not records:
        raise ValueError("官方返回中没有找到历史行情记录")
    frame = pd.DataFrame(records)
    date_col = next((c for c in frame.columns if str(c).lower() in {"date", "tradedate", "trade_date", "日期", "交易日期"}), None)
    price_col = next((c for c in frame.columns if str(c).lower() in {"close", "closeprice", "indexpoint", "收盘", "收盘价", "收盘点位"}), None)
    if date_col is None:
        date_col = next((c for c in frame.columns if "date" in str(c).lower() or "日期" in str(c)), None)
    if price_col is None:
        price_col = next((c for c in frame.columns if "close" in str(c).lower() or "收盘" in str(c) or "price" in str(c).lower()), None)
    if date_col is None or price_col is None:
        raise ValueError(f"无法识别日期/收盘价字段，可见字段: {', '.join(map(str, frame.columns))}")
    result = pd.DataFrame({"date": pd.to_datetime(frame[date_col], errors="coerce"), "price": frame[price_col].map(_number)})
    result = result.dropna().drop_duplicates("date").sort_values("date")
    if len(result) < 2:
        raise ValueError("官方返回的有效历史数据不足 2 个交易日")
    return result.set_index("date")["price"].rename("price")


def _frame_from_eastmoney_payload(payload: Any) -> pd.Series:
    data = payload.get("data") if isinstance(payload, dict) else None
    rows = data.get("klines") if isinstance(data, dict) else None
    if not rows:
        raise ValueError("东方财富返回中没有历史行情记录")
    records = [row.split(",") for row in rows]
    frame = pd.DataFrame({"date": [row[0] for row in records], "price": [row[2] for row in records]})
    result = pd.DataFrame({"date": pd.to_datetime(frame["date"], errors="coerce", format="mixed"), "price": frame["price"].map(_number)})
    result = result.dropna().drop_duplicates("date").sort_values("date")
    if len(result) < 2:
        raise ValueError("东方财富返回的有效数据不足 2 个交易日")
    return result.set_index("date")["price"].rename("price")


def _frame_from_sina_payload(payload: Any) -> pd.Series:
    if not isinstance(payload, list) or not payload:
        raise ValueError("新浪返回中没有历史行情记录")
    frame = pd.DataFrame(payload)
    if "day" not in frame.columns or "close" not in frame.columns:
        raise ValueError(f"无法识别新浪日期/收盘价字段: {list(frame.columns)}")
    result = pd.DataFrame({"date": pd.to_datetime(frame["day"], errors="coerce", format="mixed"), "price": frame["close"].map(_number)})
    result = result.dropna().drop_duplicates("date").sort_values("date")
    if len(result) < 2:
        raise ValueError("新浪返回的有效数据不足 2 个交易日")
    return result.set_index("date")["price"].rename("price")


def _official_get(url: str, params: dict[str, Any]) -> Any:
    response = requests.get(url, params=params, timeout=20, headers={"User-Agent": "Mozilla/5.0 portfolio-tools"})
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "json" not in content_type:
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise ValueError("官网返回的不是 JSON 数据，可能需要更新接口适配器") from exc
    return response.json()


def _cache_path(spec: IndexSpec) -> Path:
    return CACHE_DIR / f"{spec.provider}_{spec.code}.csv"


def _read_price_cache(spec: IndexSpec) -> pd.Series:
    path = _cache_path(spec)
    if not path.exists():
        return pd.Series(dtype="float64", name="price")
    cached = pd.read_csv(path, parse_dates=["date"])
    if not {"date", "price"}.issubset(cached.columns):
        return pd.Series(dtype="float64", name="price")
    return (
        cached.dropna(subset=["date", "price"])
        .drop_duplicates("date")
        .set_index("date")["price"]
        .sort_index()
        .rename("price")
    )


def _write_price_cache(spec: IndexSpec, prices: pd.Series) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    output = prices.rename("price").rename_axis("date").reset_index()
    path = _cache_path(spec)
    temporary_path = path.with_suffix(".tmp")
    output.to_csv(temporary_path, index=False, date_format="%Y-%m-%d")
    temporary_path.replace(path)


def _fetch_official_remote(spec: IndexSpec, start: date, end: date) -> pd.Series:
    start_text = start.strftime("%Y-%m-%d")
    end_text = end.strftime("%Y-%m-%d")
    if spec.provider == "东方财富ETF":
        candidates = [
            ("https://push2his.eastmoney.com/api/qt/stock/kline/get", {"secid": f"1.{spec.code}", "ut": "fa5fd1943c7b386f172d6893dbfba10b", "fields1": "f1,f2,f3,f4,f5,f6", "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61", "klt": "101", "fqt": "1", "beg": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d")}, _frame_from_eastmoney_payload),
            ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData", {"symbol": f"sh{spec.code}", "scale": "240", "ma": "no", "datalen": "6000"}, _frame_from_sina_payload),
        ]
    elif spec.provider == "中证指数":
        candidates = [
            ("https://www.csindex.com.cn/csindex-home/perf/index-perf", {"indexCode": spec.code, "startDate": start_text, "endDate": end_text}, _frame_from_payload),
        ]
    else:
        candidates = [
            ("https://www.cnindex.com.cn/api/quote", {"indexCode": spec.code, "startDate": start_text, "endDate": end_text}, _frame_from_payload),
            ("https://www.cnindex.com.cn/api/index/history", {"indexCode": spec.code, "startDate": start_text, "endDate": end_text}, _frame_from_payload),
        ]
    errors = []
    for url, params, parse in candidates:
        try:
            return parse(_official_get(url, params))
        except (requests.RequestException, ValueError, KeyError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("; ".join(errors))


def fetch_official(spec: IndexSpec, start: date, end: date) -> pd.Series:
    """Read a covered range from disk, fetching and merging only missing dates."""
    requested_start = pd.Timestamp(start)
    requested_end = pd.Timestamp(end)
    cached = _read_price_cache(spec)
    if not cached.empty:
        cache_start = cached.index.min()
        cache_end = cached.index.max()
        if cache_start <= requested_start and cache_end >= requested_end:
            return cached.loc[requested_start:requested_end]

    fetch_start = start
    fetch_end = end
    if not cached.empty:
        if cached.index.min() <= requested_start:
            fetch_start = (cached.index.max() + pd.Timedelta(days=1)).date()
        if cached.index.max() >= requested_end:
            fetch_end = (cached.index.min() - pd.Timedelta(days=1)).date()
    if fetch_start <= fetch_end:
        fresh = _fetch_official_remote(spec, fetch_start, fetch_end)
        cached = pd.concat([cached, fresh]).sort_index()
        cached = cached[~cached.index.duplicated(keep="last")]
        _write_price_cache(spec, cached)
    return cached.loc[requested_start:requested_end]


def parse_uploaded_csv(uploaded: Any) -> pd.Series:
    frame = pd.read_csv(uploaded)
    if frame.shape[1] < 2:
        raise ValueError("CSV 至少需要两列：日期、价格")
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    date_col = next((c for c in frame.columns if c in {"date", "日期", "tradedate"}), frame.columns[0])
    price_col = next((c for c in frame.columns if c in {"price", "close", "收盘价", "收盘"}), frame.columns[1])
    result = pd.DataFrame({"date": pd.to_datetime(frame[date_col], errors="coerce"), "price": frame[price_col].map(_number)})
    result = result.dropna().drop_duplicates("date").sort_values("date")
    if len(result) < 2:
        raise ValueError("CSV 中有效价格数据不足 2 个交易日")
    return result.set_index("date")["price"].rename("price")


def annual_return(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    return float((1 + returns).prod() ** (252 / len(returns)) - 1)


def portfolio_metrics(returns: pd.Series, risk_free: float) -> dict[str, float]:
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    ann_return = annual_return(returns)
    ann_vol = float(returns.std(ddof=1) * np.sqrt(252))
    sharpe = (ann_return - risk_free) / ann_vol if ann_vol > 0 else np.nan
    max_drawdown = float(drawdown.min())
    calmar = ann_return / abs(max_drawdown) if max_drawdown < 0 else np.nan
    return {"年化收益": ann_return, "年化波动": ann_vol, "夏普比率": sharpe, "最大回撤": max_drawdown, "卡玛比率": calmar}


def _normalize(weights: np.ndarray) -> np.ndarray:
    weights = np.clip(weights, 0, None)
    total = weights.sum()
    return weights / total if total else np.ones_like(weights) / len(weights)


def _risk_parity_weights(sample: pd.DataFrame) -> np.ndarray:
    covariance = sample.cov().values * 252
    count = len(sample.columns)
    initial = np.ones(count) / count

    def objective(weights: np.ndarray) -> float:
        portfolio_volatility = np.sqrt(max(float(weights @ covariance @ weights), 1e-12))
        contribution = weights * (covariance @ weights) / portfolio_volatility
        target = np.full(count, contribution.sum() / count)
        return float(np.sum((contribution - target) ** 2))

    if minimize is None:
        return initial
    result = minimize(objective, initial, method="SLSQP", bounds=[(0, 1)] * count, constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1})
    return _normalize(result.x if result.success else initial)


def _risk_budget_weights(sample: pd.DataFrame, budgets: dict[str, float] | None) -> np.ndarray:
    covariance = sample.cov().values * 252
    count = len(sample.columns)
    initial = np.ones(count) / count
    limits = np.array([budgets.get(name, 1 / count) for name in sample.columns], dtype=float) if budgets else np.ones(count) / count
    if np.any(limits <= 0):
        raise ValueError("风险预算必须全部大于 0")

    def contribution(weights: np.ndarray) -> np.ndarray:
        variance = max(float(weights @ covariance @ weights), 1e-12)
        return weights * (covariance @ weights) / variance

    if minimize is None:
        return initial
    constraints = [{"type": "eq", "fun": lambda weights: weights.sum() - 1}, {"type": "ineq", "fun": lambda weights: limits - contribution(weights)}]
    result = minimize(lambda weights: float(weights @ covariance @ weights), initial, method="SLSQP", bounds=[(0, 1)] * count, constraints=constraints)
    return _normalize(result.x if result.success else initial)


def _mpt_weights(sample: pd.DataFrame, objective: str, target: float) -> np.ndarray:
    annual_mean = sample.mean().values * 252
    annual_cov = sample.cov().values * 252
    count = len(sample.columns)
    initial = np.ones(count) / count
    volatility = lambda weights: float(np.sqrt(max(weights @ annual_cov @ weights, 1e-12)))
    if minimize is None:
        return initial
    if objective == "max-return-at-volatility":
        constraints = [{"type": "eq", "fun": lambda weights: weights.sum() - 1}, {"type": "ineq", "fun": lambda weights: target - volatility(weights)}]
        target_objective = lambda weights: -float(weights @ annual_mean)
    else:
        constraints = [{"type": "eq", "fun": lambda weights: weights.sum() - 1}, {"type": "ineq", "fun": lambda weights: float(weights @ annual_mean) - target}]
        target_objective = volatility
    result = minimize(target_objective, initial, method="SLSQP", bounds=[(0, 1)] * count, constraints=constraints)
    return _normalize(result.x if result.success else initial)


def strategy_weights(
    returns: pd.DataFrame,
    strategy: str,
    lookback: int,
    risk_free: float = 0.02,
    optimize_target: str = "最大夏普",
    mpt_objective: str = "max-return-at-volatility",
    mpt_target: float = 0.20,
    risk_budgets: dict[str, float] | None = None,
) -> pd.Series:
    sample = returns.tail(max(20, min(lookback, len(returns))))
    covariance = sample.cov().values * 252
    n_assets = returns.shape[1]
    effective_strategy = optimize_target if strategy == "自动最优组合" else strategy
    if effective_strategy == "等权组合":
        weights = np.ones(n_assets) / n_assets
    elif effective_strategy == "反波动率":
        weights = 1 / sample.std().replace(0, np.nan).fillna(sample.std().mean()).values
        weights = _normalize(weights)
    elif effective_strategy == "动量组合":
        momentum = (1 + sample).prod() - 1
        weights = _normalize(np.maximum(momentum.values, 0.01))
    elif effective_strategy == "风险平价":
        weights = _risk_parity_weights(sample)
    elif effective_strategy == "风险预算":
        weights = _risk_budget_weights(sample, risk_budgets)
    elif effective_strategy == "MPT均值-方差":
        weights = _mpt_weights(sample, mpt_objective, mpt_target)
    elif effective_strategy == "最小方差" and minimize is not None:
        objective = lambda w: float(w @ covariance @ w)
        result = minimize(objective, np.ones(n_assets) / n_assets, bounds=[(0, 1)] * n_assets, constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
        weights = _normalize(result.x if result.success else np.ones(n_assets))
    elif effective_strategy == "最大夏普" and minimize is not None:
        mean = sample.mean().values * 252
        objective = lambda w: -float((w @ mean - risk_free) / np.sqrt(max(w @ covariance @ w, 1e-12)))
        result = minimize(objective, np.ones(n_assets) / n_assets, bounds=[(0, 1)] * n_assets, constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
        weights = _normalize(result.x if result.success else np.ones(n_assets))
    else:
        weights = np.ones(n_assets) / n_assets
    return pd.Series(weights, index=returns.columns)


def strategy_label(strategy: str, optimize_target: str = "最大夏普") -> str:
    label = f"自动最优组合（{optimize_target}）" if strategy == "自动最优组合" else strategy
    return label + ("（当前环境未安装 scipy，已回退等权）" if strategy in {"最小方差", "最大夏普", "自动最优组合"} and minimize is None else "")


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Noto+Sans+SC:wght@400;600;700;900&display=swap');
:root { --ink:#17221d; --muted:#68756d; --paper:#f4f0e7; --accent:#da5b38; --green:#1f6a56; }
.stApp { background: radial-gradient(circle at 85% 5%, #dce8dc 0, transparent 30%), var(--paper); color:var(--ink); }
html, body, [class*="css"] { font-family:'Noto Sans SC', sans-serif; }
.block-container { padding-top:2rem; max-width:1280px; }
.hero { border-bottom:1px solid #b9c4b9; padding: 1.4rem 0 1.2rem; margin-bottom:1.2rem; }
.eyebrow { color:var(--accent); font-family:'DM Mono'; letter-spacing:.12em; font-size:.76rem; }
.hero h1 { font-size: clamp(2rem, 4vw, 4.5rem); line-height:1; letter-spacing:-.04em; margin:.5rem 0; font-weight:900; }
.hero p { color:var(--muted); max-width:700px; margin:0; }
.metric-card { background:#e9eee6; border:1px solid #cad5c8; padding:1rem; min-height:115px; }
.metric-name { font-size:.76rem; color:var(--muted); }
.metric-value { font-family:'DM Mono'; font-size:1.65rem; color:var(--green); margin-top:.5rem; }
[data-testid="stSidebar"] { background:#e7ede4; border-right:1px solid #c5d0c3; }
</style>
<div class="hero"><div class="eyebrow">INDEX PORTFOLIO / RESEARCH DESK</div><h1>指数组合实验室</h1><p>从中证指数、国证指数官网读取历史价格，比较不同组合策略的收益与风险。</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 参数面板")
    selected_names = st.multiselect("选择指数", [f"{x.code} · {x.name}" for x in INDEX_CATALOG], default=["000300 · 沪深300", "000905 · 中证500", "399006 · 创业板指"])
    start = st.date_input("开始日期", date.today() - timedelta(days=365 * 5))
    end = st.date_input("结束日期", date.today())
    risk_free = st.number_input("年化无风险利率", min_value=0.0, max_value=0.2, value=0.02, step=0.005, format="%.3f")
    lookback = st.slider("策略观察窗口（交易日）", 20, 756, 126, 21)
    optimize_target = st.selectbox("自动最优组合目标", ["最大夏普", "最小方差"], help="最大夏普追求风险调整后收益，最小方差追求组合波动率最低。")
    mpt_objective_label = st.selectbox("MPT有效前沿目标", ["给定波动率最大化收益", "给定收益率最小化波动率"])
    mpt_objective = "max-return-at-volatility" if mpt_objective_label == "给定波动率最大化收益" else "min-volatility-at-return"
    mpt_target_label = "目标年化波动率" if mpt_objective == "max-return-at-volatility" else "目标年化收益率"
    mpt_target = st.number_input(mpt_target_label, min_value=0.01, max_value=2.0, value=0.20, step=0.01, format="%.2f")
    risk_budget_text = st.text_area("风险贡献上限 JSON", value="", placeholder='{"沪深300": 0.30, "中证500": 0.30}', help="键为资产名称，值为该资产允许占用的组合风险贡献上限。留空时使用等权风险上限。")
    try:
        risk_budgets = json.loads(risk_budget_text) if risk_budget_text.strip() else None
        if risk_budgets is not None and not isinstance(risk_budgets, dict):
            raise ValueError("必须是 JSON 对象")
    except (json.JSONDecodeError, ValueError) as exc:
        st.error(f"风险预算 JSON 无效：{exc}")
        risk_budgets = None
    strategies = st.multiselect("组合策略", ["自动最优组合", "风险预算", "MPT均值-方差", "风险平价", "等权组合", "反波动率", "动量组合", "最小方差", "最大夏普"], default=["自动最优组合", "风险预算", "风险平价"])
    uploaded = st.file_uploader("可选：上传 CSV 兜底", type="csv", help="格式：date,price,index_code；上传后会按 index_code 合并。")
    run = st.button("开始分析", type="primary", use_container_width=True)

if not selected_names:
    st.info("请至少选择一个指数。")
    st.stop()
if start >= end:
    st.error("开始日期必须早于结束日期。")
    st.stop()

if run or "prices" not in st.session_state:
    specs = [next(item for item in INDEX_CATALOG if f"{item.code} · {item.name}" == value) for value in selected_names]
    prices: dict[str, pd.Series] = {}
    messages = []
    upload_frame = None
    if uploaded is not None:
        try:
            upload_frame = pd.read_csv(uploaded)
            upload_frame["date"] = pd.to_datetime(upload_frame["date"], errors="coerce")
            upload_frame["price"] = upload_frame["price"].map(_number)
        except Exception as exc:
            messages.append(f"CSV 读取失败：{exc}")
    progress = st.progress(0, text="正在连接官网数据源...")
    for index, spec in enumerate(specs, start=1):
        try:
            if upload_frame is not None and "index_code" in upload_frame and spec.code in upload_frame["index_code"].astype(str).str.zfill(6).values:
                subset = upload_frame[upload_frame["index_code"].astype(str).str.zfill(6) == spec.code].dropna(subset=["date", "price"])
                prices[spec.name] = subset.set_index("date")["price"].sort_index()
            else:
                prices[spec.name] = fetch_official(spec, start, end)
        except Exception as exc:
            messages.append(f"{spec.name}（{spec.provider}）获取失败：{exc}")
        progress.progress(index / len(specs), text=f"已处理 {index}/{len(specs)} 个指数")
    progress.empty()
    st.session_state.prices = prices
    st.session_state.messages = messages
    st.session_state.specs = specs

prices = st.session_state.get("prices", {})
messages = st.session_state.get("messages", [])
if messages:
    for message in messages:
        st.warning(message)
if len(prices) < 1:
    st.error("没有可用于分析的价格序列。请检查官网访问、接口变更，或上传 CSV。")
    st.stop()

price_frame = pd.concat(prices, axis=1).dropna(how="all").ffill().dropna()
returns = price_frame.pct_change().dropna()
if returns.empty:
    st.error("价格序列重叠交易日不足，无法计算收益率。")
    st.stop()

st.markdown(f"### 数据概览  ·  {price_frame.index.min().date()} 至 {price_frame.index.max().date()}")
st.line_chart(price_frame / price_frame.iloc[0] * 100, height=300)

results = []
weights_table = []
for strategy in strategies:
    weights = strategy_weights(returns, strategy, lookback, risk_free, optimize_target, mpt_objective, mpt_target, risk_budgets)
    portfolio_returns = returns @ weights
    metrics = portfolio_metrics(portfolio_returns, risk_free)
    metrics["策略"] = strategy_label(strategy, optimize_target)
    results.append(metrics)
    weights_table.append(pd.DataFrame({"策略": strategy_label(strategy, optimize_target), "指数": weights.index, "权重": weights.values}))

if not strategies:
    st.info("请在左侧至少选择一个策略。")
    st.stop()

result_frame = pd.DataFrame(results).set_index("策略")
cols = st.columns(5)
for col, metric in zip(cols, ["年化收益", "年化波动", "夏普比率", "最大回撤", "卡玛比率"]):
    value = result_frame[metric].iloc[0]
    rendered = "—" if pd.isna(value) else (f"{value:.2%}" if metric in {"年化收益", "年化波动", "最大回撤"} else f"{value:.2f}")
    col.markdown(f'<div class="metric-card"><div class="metric-name">{metric}</div><div class="metric-value">{rendered}</div></div>', unsafe_allow_html=True)

st.markdown("### 策略对比")
st.dataframe(result_frame.style.format({"年化收益":"{:.2%}", "年化波动":"{:.2%}", "夏普比率":"{:.2f}", "最大回撤":"{:.2%}", "卡玛比率":"{:.2f}"}), use_container_width=True)

wealth = pd.DataFrame(index=returns.index)
for strategy in strategies:
    weights = strategy_weights(returns, strategy, lookback, risk_free, optimize_target, mpt_objective, mpt_target, risk_budgets)
    wealth[strategy_label(strategy, optimize_target)] = (1 + returns @ weights).cumprod()
st.line_chart(wealth, height=300)

st.markdown("### 当前组合权重")
st.dataframe(pd.concat(weights_table, ignore_index=True).pivot(index="指数", columns="策略", values="权重").style.format("{:.2%}"), use_container_width=True)

with st.expander("数据来源与口径"):
    st.markdown(f"中证指数：[官网]({CSI_HOME})；国证指数：[官网]({CNI_HOME})。收益率使用收盘点位的简单日收益率；年化按 252 个交易日；夏普比率使用年化无风险利率；卡玛比率 = 年化收益率 / 最大回撤绝对值。策略为基于整个样本期末观察窗口的一次性权重，不构成投资建议。")
