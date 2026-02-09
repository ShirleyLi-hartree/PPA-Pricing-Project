# PPA Pricing System v2

这是一个针对日本电力市场（JEPX）的 PPA（购电协议）定价与预测系统。该系统整合了多种预测模型（Prophet, XGBoost, SARIMAX, 历史平均），提供统一的 CLI 命令行接口来执行定价、预测和市场分析任务。

## 环境准备

在使用本系统前，请确保已安装 Python 3.8+ 并安装相关依赖：

```bash
pip install -r requirements.txt
```

*注意：如果还没有 `requirements.txt`，请确保安装 pandas, numpy, matplotlib, prophet, xgboost, statsmodels, scikit-learn 等核心库。*

## 命令行使用说明

系统的统一入口是 `src.cli.main` 模块。请在项目根目录下运行以下命令。

**基本语法：**

```bash
python -m src.cli.main <command> [options]
```

可用命令 (`<command>`)：
- `pricing`: 执行完整的 PPA 定价流程（训练 -> 验证 -> 定价计算）。
- `forecast`: 执行中期电价预测与回测评估。
- `analysis`: 运行市场数据分析（如基本面分析）。

---

### 1. PPA 定价 (Pricing)

`pricing` 命令用于计算特定合同期内的固定电价（Fixed Price）。它会自动进行滚动窗口训练、模型验证，并根据预测结果计算不同供电场景下的 PPA 价格。

**常用参数：**
- `--region`: 目标区域 (tokyo, kansai, etc.)，默认 `tokyo`。
- `--models`: 使用的模型列表，可选 `xgboost`, `prophet`, `sarimax`, `historical`。默认使用 historical 和 sarimax。
- `--contract-start`: 合同开始日期 (YYYY-MM-DD)，默认 `2027-04-01`。
- `--contract-end`: 合同结束日期 (YYYY-MM-DD)，默认 `2028-03-31`。
- `--volume`: 总签约电量 (MWh)，默认 `2.0`。
- `--train-years`: 训练数据使用的年数，默认 `3`。
- `--skip-validation`: 跳过验证步骤，仅计算价格。

**运行示例：**

1.  **基础运行（使用默认参数）**：
    ```bash
    python -m src.cli.main pricing
    ```

2.  **指定区域和合同期**：
    计算关西电力（Kansai）2025年度的 PPA 价格：
    ```bash
    python -m src.cli.main pricing --region kansai --contract-start 2025-04-01 --contract-end 2026-03-31
    ```

3.  **使用高级模型（XGBoost 和 Prophet）**：
    *注意：XGBoost 和 Prophet 计算时间较长。*
    ```bash
    python -m src.cli.main pricing --models xgboost prophet --region tokyo
    ```

4.  **自定义电量和训练窗口**：
    ```bash
    python -m src.cli.main pricing --volume 1000 --train-years 5
    ```

---

### 2. 电价预测与回测 (Forecast)

`forecast` 命令专注于模型性能评估和中期预测。它会执行滚动回测（Rolling Backtest）来评估模型在不同时间段的表现。

**常用参数：**
- `--region`: 目标区域，默认 `tokyo`。
- `--forecast-days`: 预测展望期（天数），默认 `90`。
- `--years`: 训练数据年数，默认 `3`。
- `--step-days`: 回测滑动的步长（天数），默认 `30`。
- `--skip-prophet`: 跳过 Prophet 模型（以加快运行速度）。

**运行示例：**

1.  **运行 90 天预测回测**：
    ```bash
    python -m src.cli.main forecast --forecast-days 90
    ```

2.  **快速回测（跳过 Prophet）**：
    ```bash
    python -m src.cli.main forecast --skip-prophet
    ```

---

### 3. 数据分析 (Analysis)

`analysis` 命令用于生成市场分析报告。

**示例：**

```bash
python -m src.cli.main analysis --type fundamental --region kansai
```

---

## 输出结果

所有运行结果默认保存在 `results/` 目录下，包含：
- 预测数据 CSV
- 模型评估报告
- PPA 定价结果日志

## 常见问题

**Q: 找不到模块 `src`？**
A: 请确保你在项目的根目录下运行命令（即 `src` 文件夹所在的上一级目录）。

**Q: 数据从哪里加载？**
A: 默认从 `data/` 目录加载 `spot_summary_*.csv` 文件。请确保该目录下有 JEPX 的历史数据文件。