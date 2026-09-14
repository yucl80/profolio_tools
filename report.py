from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd

from backtest import INDEXES, _read_cache

TMP_DIR = Path(__file__).resolve().parent / "tmp"


DEFAULT_MODELS = {
    "test_max_sharpe": "最大夏普",
    "test_risk_parity": "风险平价",
    "test_mpt_vol": "MPT：目标波动率 20%",
    "test_mpt_return": "MPT：目标收益率 12%",
}

REQUIRED_METRICS = ["年化收益率", "年化波动率", "夏普比率", "最大回撤"]


def read_model(directory: Path, name: str) -> tuple[dict[str, float | str], pd.DataFrame]:
    metrics_path = directory / "metrics.csv"
    weights_path = directory / "latest_weights.csv"
    curve_path = directory / "portfolio_curve.csv"
    if not metrics_path.exists() or not weights_path.exists() or not curve_path.exists():
        raise FileNotFoundError(f"{directory} 缺少 metrics.csv、latest_weights.csv 或 portfolio_curve.csv")
    metrics = pd.read_csv(metrics_path)
    weights = pd.read_csv(weights_path)
    curve = pd.read_csv(curve_path, parse_dates=["date"])
    missing = [column for column in REQUIRED_METRICS if column not in metrics.columns]
    if missing:
        raise ValueError(f"{metrics_path} 缺少字段: {', '.join(missing)}")
    if not {"index", "weight_pct"}.issubset(weights.columns):
        raise ValueError(f"{weights_path} 缺少 index 或 weight_pct 字段")
    if curve.empty or "date" not in curve.columns:
        raise ValueError(f"{curve_path} 没有有效回测日期")
    summary = {"模型名称": name}
    summary["回测开始日期"] = curve["date"].min().strftime("%Y-%m-%d")
    summary["回测结束日期"] = curve["date"].max().strftime("%Y-%m-%d")
    summary.update({column: float(metrics.iloc[0][column]) for column in REQUIRED_METRICS})
    return summary, weights[["index", "weight_pct"]].copy()


def pct(value: float) -> str:
    return f"{value:.2%}"


def asset_period_metrics(prices: pd.Series, end_date: pd.Timestamp, years: int) -> dict[str, float | str]:
    prices = prices.loc[:end_date].dropna()
    if len(prices) < 2:
        return {"区间": f"近{years}年（数据不足）", "年化收益率": float("nan"), "年化波动率": float("nan"), "最大回撤": float("nan")}
    target_start = end_date - pd.DateOffset(years=years)
    period_prices = prices.loc[target_start:end_date]
    if len(period_prices) < 2:
        period_prices = prices
    returns = period_prices.pct_change().dropna()
    elapsed_years = max((period_prices.index[-1] - period_prices.index[0]).days / 365.25, 1 / 365.25)
    annual_return = float((period_prices.iloc[-1] / period_prices.iloc[0]) ** (1 / elapsed_years) - 1)
    annual_volatility = float(returns.std(ddof=1) * (252 ** 0.5)) if len(returns) > 1 else float("nan")
    wealth = (1 + returns).cumprod()
    max_drawdown = float((wealth / wealth.cummax() - 1).min())
    actual_start = period_prices.index[0].strftime("%Y-%m-%d")
    actual_end = period_prices.index[-1].strftime("%Y-%m-%d")
    label = f"近{years}年（{actual_start} 至 {actual_end}）"
    return {"区间": label, "年化收益率": annual_return, "年化波动率": annual_volatility, "最大回撤": max_drawdown}


def build_asset_history(model_weights: list[tuple[str, pd.DataFrame]], end_date: pd.Timestamp) -> pd.DataFrame:
    asset_names = sorted({str(row["index"]) for _, weights in model_weights for _, row in weights.iterrows()})
    rows = []
    for asset_name in asset_names:
        index = next((item for item in INDEXES.values() if item.name == asset_name), None)
        if index is None:
            continue
        prices = next((_read_cache(index, code) for code in index.total_return_codes if not _read_cache(index, code).empty), _read_cache(index))
        for years in (3, 5, 10):
            metrics = asset_period_metrics(prices, end_date, years)
            rows.append({"资产": asset_name, **metrics})
    return pd.DataFrame(rows)


def build_report(model_dirs: dict[str, str], output: Path) -> None:
    summaries = []
    model_weights: list[tuple[str, pd.DataFrame]] = []
    for directory_name, model_name in model_dirs.items():
        summary, weights = read_model(Path(directory_name), model_name)
        summaries.append(summary)
        model_weights.append((model_name, weights))

    summary_frame = pd.DataFrame(summaries)
    summary_frame["最大回撤"] = summary_frame["最大回撤"].abs()
    summary_frame = summary_frame[["模型名称", "回测开始日期", "回测结束日期", "年化收益率", "年化波动率", "夏普比率", "最大回撤"]]
    summary_html = summary_frame.to_html(
        index=False,
        escape=True,
        formatters={
            "年化收益率": pct,
            "年化波动率": pct,
            "夏普比率": lambda value: f"{value:.4f}",
            "最大回撤": pct,
        },
        classes="summary-table",
    )

    detail_sections = []
    for model_name, weights in model_weights:
        weights = weights.sort_values("weight_pct", ascending=False)
        rows = []
        for _, row in weights.iterrows():
            rows.append(
                f"<tr><td>{html.escape(str(row['index']))}</td>"
                f"<td class=\"weight\">{float(row['weight_pct']):.2f}%</td></tr>"
            )
        detail_sections.append(
            f"<section class=\"model-card\"><h3>{html.escape(model_name)}</h3>"
            f"<table><thead><tr><th>资产</th><th>组合比例</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></section>"
        )

    report_end = pd.Timestamp(summary_frame["回测结束日期"].max())
    asset_history = build_asset_history(model_weights, report_end)
    asset_history_tables = []
    for years in (3, 5, 10):
        period_history = asset_history[asset_history["区间"].str.startswith(f"近{years}年")].copy()
        period_history["区间"] = period_history["区间"].str.replace(f"近{years}年（", "", regex=False).str.rstrip("）")
        period_history = period_history.rename(columns={"区间": "实际数据区间"})
        table_html = period_history.to_html(
            index=False,
            escape=True,
            formatters={
                "年化收益率": lambda value: "—" if pd.isna(value) else pct(value),
                "年化波动率": lambda value: "—" if pd.isna(value) else pct(value),
                "最大回撤": lambda value: "—" if pd.isna(value) else pct(value),
            },
            classes="summary-table",
        )
        asset_history_tables.append(f"<h3>近{years}年</h3>{table_html}")
    asset_history_html = "".join(asset_history_tables)

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>指数组合模型回测报告</title>
<style>
:root {{ --ink:#17221d; --muted:#68756d; --paper:#f4f0e7; --green:#1f6a56; --accent:#da5b38; --line:#cbd5c9; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:radial-gradient(circle at 90% 0,#dce8dc,transparent 34%),var(--paper); color:var(--ink); font-family:"Segoe UI","Microsoft YaHei",sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:42px 24px 64px; }}
.eyebrow {{ color:var(--accent); font:600 12px Consolas,monospace; letter-spacing:.14em; }}
h1 {{ margin:10px 0 8px; font-size:clamp(30px,5vw,58px); letter-spacing:-.04em; }}
.subtitle {{ color:var(--muted); margin:0 0 34px; }}
h2 {{ margin:34px 0 14px; font-size:22px; }}
h3 {{ margin:0 0 14px; font-size:18px; }}
table {{ width:100%; border-collapse:collapse; background:rgba(255,255,255,.52); }}
th,td {{ padding:12px 14px; border-bottom:1px solid var(--line); text-align:left; }}
th {{ color:var(--muted); font-size:13px; font-weight:600; }}
td {{ font-variant-numeric:tabular-nums; }}
.summary-table td:not(:first-child), .weight {{ color:var(--green); font-weight:700; }}
.models {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:18px; }}
.model-card {{ border:1px solid var(--line); background:rgba(233,238,230,.72); padding:18px; }}
.model-card table {{ background:transparent; }}
.note {{ color:var(--muted); font-size:13px; margin-top:28px; }}
@media (max-width:700px) {{ th,td {{ padding:10px 8px; font-size:13px; }} main {{ padding:28px 14px 48px; }} }}
</style>
</head>
<body><main>
<div class="eyebrow">INDEX PORTFOLIO / BACKTEST REPORT</div>
<h1>指数组合模型回测报告</h1>
<p class="subtitle">汇总各模型的实际回测起止日期、收益风险指标与最后一次再平衡资产比例。</p>
<h2>模型指标对比</h2>
{summary_html}
<h2>最终资产比例</h2>
<div class="models">{''.join(detail_sections)}</div>
<h2>资产历史表现</h2>
<p class="subtitle">按报告回测结束日计算各资产历史表现；区间不足时使用本地缓存可用数据。</p>
{asset_history_html}
<p class="note">最大回撤按绝对值展示；资产比例来自各模型的最后一次再平衡。报告仅用于研究与回测，不构成投资建议。</p>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(f"HTML 报告已生成: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并多个组合模型结果为 HTML 报告")
    parser.add_argument("--output", type=Path, default=TMP_DIR / "portfolio_report.html")
    parser.add_argument("--model-dir", action="append", metavar="目录=模型名称", help="自定义模型目录和名称，可重复传入")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dirs = dict(DEFAULT_MODELS)
    if args.model_dir:
        model_dirs = {}
        for item in args.model_dir:
            if "=" not in item:
                raise SystemExit("--model-dir 格式应为 目录=模型名称")
            directory, name = item.split("=", 1)
            model_dirs[directory] = name
    build_report(model_dirs, args.output)


if __name__ == "__main__":
    main()
