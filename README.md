# Machine Learning for Equity Return Prediction & Factor Modeling

**Can well-known stock characteristics predict which large-cap stocks will outperform next month, when tested honestly on data the model has never seen?**

This project predicts the next-month relative returns of 52 large-cap US stocks (2010–2026) from 8 engineered features (momentum, reversal, volatility, liquidity, market beta, idiosyncratic volatility). It compares regularized regression, gradient boosting and a PyTorch neural network in a **walk-forward** test, evaluates them with the metrics quants use (Information Coefficient, long-short portfolios), and uses PCA and clustering to uncover the latent risk factors driving the stocks.

**Status:** code complete; results section to be filled from the first full run.

## Key results

*To be filled from notebooks 02–04.*

## Why this project was rebuilt

The first version evaluated models with `train_test_split`, which **shuffles rows randomly**. The target was the next 21-day return on daily rows, so consecutive rows shared 20 of their 21 days: the test set contained near-copies of training rows. On the original data, gradient boosting scored an R² of **+0.13** with the shuffled split and **−0.48** when trained on the past and tested on the future. Notebook 02 reproduces this comparison.

This version fixes it:
- **Monthly, non-overlapping** targets (next month's return relative to the cross-sectional average)
- **Walk-forward** evaluation: expanding window, annual refits from 2015; a model only trains on months whose outcome was known at prediction time
- Hyperparameters chosen on the **most recent 20% of training months**, never on test data
- A larger universe (5 → 52 stocks) so cross-sectional rankings and factor analysis are meaningful

## Method

| Step | What happens | Notebook |
|---|---|---|
| 1. Data & features | Daily prices/volumes (Yahoo Finance); 8 features at each month-end, ranked cross-sectionally to [−0.5, 0.5] | `01_data_and_features` |
| 2. Models | Momentum baseline (no fitting), Ridge (alpha tuned on time-ordered validation), gradient boosting, PyTorch MLP (32-16, dropout, weight decay, early stopping); walk-forward 2015–2026 | `02_models_walk_forward` |
| 3. Evaluation | Out-of-sample R², monthly Spearman IC (mean, t-stat, hit rate), quintile returns, top-minus-bottom quintile long-short portfolio with 10 bps costs | `03_evaluation` |
| 4. Latent factors | PCA of daily returns (correlation matrix); PC1 vs SPY; K-means on PC2–PC6 loadings compared with true sectors (adjusted Rand index) | `04_latent_factors` |

**Features**

| Feature | Definition |
|---|---|
| `mom_12_1` | Return from 12 months ago to 1 month ago |
| `rev_1m` | Return over the last month (short-term reversal) |
| `vol_1m`, `vol_3m` | Annualized volatility of daily returns, 21 and 63 days |
| `log_dollar_vol` | Log average daily dollar volume, 21 days (liquidity) |
| `amihud` | Log Amihud illiquidity: average \|return\| / dollar volume, 21 days |
| `beta_1y` | Market beta vs SPY, 252 days |
| `idio_vol` | Residual volatility after removing market exposure, 63 days |

## Project structure

```
EquityReturnPrediction/
├── data/
│   ├── raw/            # downloaded prices and volumes (not committed)
│   └── processed/      # feature panel and predictions (not committed)
├── notebooks/
│   ├── 01_data_and_features.ipynb
│   ├── 02_models_walk_forward.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_latent_factors.ipynb
├── src/
│   ├── config.py       # paths, universe, dates, seed
│   ├── data.py         # download and cache prices/volumes
│   ├── features.py     # monthly feature panel and targets
│   ├── models.py       # Ridge, GBM, PyTorch MLP, momentum baseline, walk-forward loop
│   ├── evaluation.py   # R², IC, quintiles, long-short, leak demonstration
│   └── factors.py      # PCA factors and clustering
├── reports/            # charts and CSV summaries
├── requirements.txt
├── .env.example
└── README.md
```

## How to rerun

Python 3.11.

```bash
git clone https://github.com/Molod02/EquityReturnPrediction.git
cd EquityReturnPrediction
conda create -n equity python=3.11 -y
conda activate equity
pip install -r requirements.txt
cp .env.example .env
jupyter notebook notebooks/
```

Run the notebooks in order. The first run downloads data (about a minute) and caches it in `data/raw/`. Notebook 02 takes a few minutes (12 annual refits × 4 models). The random seed is fixed in `src/config.py`.

`numpy` is pinned below 2.0 because PyTorch 2.2, the last release for Intel Macs, is built against numpy 1.x.

## Assumptions & limitations

- **Survivorship bias:** the universe is today's large caps, all of which survived and grew since 2010. This flatters long-only returns and can bias the cross-section.
- **Small cross-section:** 52 stocks means ~10 per quintile, so long-short results are noisy.
- **Free data:** Yahoo Finance adjusted prices and volumes; no fundamentals, no point-in-time index membership.
- **Costs:** a flat 10 bps per trade; no shorting fees, market impact or taxes.
- **Settings fixed in advance:** features, universe, models and hyperparameter grids were set before looking at test results.

## Lifecycle mapping

| Stage | Code | Notebook | Output |
|---|---|---|---|
| Data & features | `src/data.py`, `src/features.py` | `01_data_and_features.ipynb` | `data/raw/*.csv`, `data/processed/panel.pkl` |
| Models | `src/models.py` | `02_models_walk_forward.ipynb` | `data/processed/predictions.pkl`, `reports/feature_importance.png` |
| Evaluation | `src/evaluation.py` | `03_evaluation.ipynb` | `reports/ic_summary.csv`, `reports/long_short*.{csv,png}` |
| Latent factors | `src/factors.py` | `04_latent_factors.ipynb` | `reports/pca_*.png` |
