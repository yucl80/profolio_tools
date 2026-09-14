# 指数组合实验室

一个基于 Streamlit 的指数组合分析工具：从中证指数、国证指数官网读取历史指数点位，并比较等权、反波动率、动量、最小方差、最大夏普等策略。

## 启动

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

打开终端显示的本地地址即可使用。

## 六指数滚动回测

新增的 `backtest.py` 支持以下资产：国证成长100（980080）、国证价值100（980081）、中证红利低波（H30269）、中证科创成长（000690）、国证创业板成长（399667）、国证自由现金流（980092）以及黄金ETF华安（518880）。黄金ETF优先取东方财富场内 ETF 行情接口，该接口不可达时自动回退到新浪财经日线接口。两个渠道都是 ETF 收盘价，不属于中证或国证指数，不适用指数全收益代码规则；新浪渠道可回溯到 2013 年上市初期。如果官网提供对应全收益指数，程序优先使用全收益指数并单独缓存，否则自动回退到价格指数。当前已配置的全收益代码包括：480080、480081、H20269、480092。默认使用全部资产，也可以只选择部分资产：

```powershell
python backtest.py --indices 成长100 价值100 红利低波 科创成长 创业板成长 自由现金流 --start 2018-01-01 --end 2026-09-14
```

例如使用部分指数进行回测：

```powershell
python backtest.py --indices 成长100 价值100 红利低波 --start 2015-01-05 --end 2025-12-31
```

程序按 `lookback` 个交易日滚动训练，每 `rebalance` 个交易日再平衡一次。默认优化目标为纯最大化夏普比率；也支持 `--model risk-parity` 风险平价模型，使各指数对组合总风险的贡献尽量相等。如需额外提高收益率权重，可通过 `--return-weight` 调整。运行产生的缓存和结果默认写入 `tmp/`，例如 `tmp/backtest_output/metrics.csv`、`weights.csv`、`latest_weights.csv` 和 `portfolio_curve.csv`。其中 `latest_weights.csv` 保存最后一次再平衡的资产比例，并附带该组合的年化收益率、年化波动率和夏普比率；`weights.csv` 保存全部历史再平衡比例。

例如只测试价值和成长指数，并每月再平衡：

```powershell
python backtest.py --indices 成长100 价值100 --lookback 252 --rebalance 21 --risk-free 0.02
```

风险平价回测：

```powershell
python backtest.py --model risk-parity --indices 成长100 价值100 红利低波 科创成长 创业板成长 自由现金流
```

MPT 均值-方差回测：

```powershell
# 给定年化波动率 20%，最大化预期收益
python backtest.py --model mpt --mpt-objective max-return-at-volatility --mpt-target 0.20

# 给定年化收益率 12%，最小化组合波动率
python backtest.py --model mpt --mpt-objective min-volatility-at-return --mpt-target 0.12
```

风险预算回测：

```powershell
python backtest.py --model risk-budget --risk-budgets '{"成长100":0.30,"价值100":0.30,"红利低波":0.30,"科创成长":0.30,"创业板成长":0.30,"自由现金流":0.30}'
```

风险预算值是各资产对组合总风险的贡献上限；留空时使用等权上限。

## 合并 HTML 报告

四个模型完成回测后，运行以下命令，将模型名称、实际回测起止日期、年化收益率、年化波动率、夏普比率、最大回撤和最终资产比例合并到一个 HTML 文件：

```powershell
python report.py --output tmp/portfolio_report.html
```

默认读取 `test_max_sharpe`、`test_risk_parity`、`test_mpt_vol` 和 `test_mpt_return` 四个目录。也可以自定义目录和模型名称：

```powershell
python report.py --model-dir risk_parity_output=风险平价 --model-dir mpt_volatility_output=MPT目标波动率
```

## 数据格式兜底

如果官网接口调整或网络受限，可上传 CSV：

```csv
date,price,index_code
2024-01-02,3429.68,000300
2024-01-03,3440.70,000300
```

CSV 的 `date`、`price`、`index_code` 为必需字段。程序优先使用上传文件中对应指数的数据，否则访问对应官网适配器。

## 计算口径

- 日收益率：`P_t / P_(t-1) - 1`
- 年化收益率：几何年化，按 252 个交易日
- 年化波动率：日收益标准差 × `sqrt(252)`
- 夏普比率：`(年化收益率 - 无风险利率) / 年化波动率`
- 卡玛比率：`年化收益率 / |最大回撤|`

程序仅用于研究与回测，不构成投资建议。官网接口属于对方网站公开服务，若其接口版本变化，请在 `fetch_official` 中更新候选地址与字段映射。
