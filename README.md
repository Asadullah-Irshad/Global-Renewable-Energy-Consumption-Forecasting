# Global Renewable Energy Consumption Forecasting

### A Comparative Benchmarking Study of Statistical, Machine Learning, and Deep Learning Models

> Benchmarking ARIMA, XGBoost, LSTM & Transformer for global renewable energy forecasting (World Bank, 1960–2020). Code for our CEIS 2026 paper.

This repository contains the code (Jupyter notebook) accompanying the published paper:

> **Biswas, S., Irshad, A., & Roy, P. (2026).** Global Renewable Energy Consumption Forecasting: A Comparative Benchmarking Study of Statistical, Machine Learning, and Deep Learning Models. *Computer Engineering and Intelligent Systems*, 17(1), 44–57.
>
> **DOI:** [10.7176/CEIS/17-1-05](https://doi.org/10.7176/CEIS/17-1-05)
> **Article:** https://iiste.org/Journals/index.php/CEIS/article/view/63772
> **Published:** March 28, 2026 · ISSN (Online) 2222-2863

## Abstract

Accurate forecasting of renewable energy consumption is essential for energy policy planning, infrastructure investment, and monitoring the global energy transition. This study presents a rigorous comparative benchmarking of four forecasting approaches — ARIMA, XGBoost, LSTM, and Transformer — applied to annual renewable energy consumption data from the World Bank (`EG.FEC.RNEW.ZS` indicator, 1960–2020) across 11 aggregate regions and income groups. Each model undergoes automated hyperparameter optimisation, and predictive accuracy is evaluated on a held-out test period using RMSE. Deep learning models outperform classical baselines: LSTM achieves the best test RMSE (0.7286), followed by Transformer (0.8938), ARIMA (1.2294), and XGBoost (1.2518). The champion LSTM model is retrained per region to generate 20-year forecasts (2021–2040).

## Models benchmarked

ARIMA · XGBoost · RNN · GRU · LSTM · Transformer · CNN-LSTM · ARIMA-LSTM · XGB-LSTM

## Repository structure

```
Global-Renewable-Energy-Consumption-Forecasting/
├── forecasting_benchmark_v2_1.ipynb   # Full analysis notebook
├── Src/                              # Plain-Python (.py) version of the notebook
├── requirements.txt                   # Python dependencies
├── README.md
├── LICENSE
├── CITATION.cff                      # Machine-readable citation ("Cite this repository")
├── Data/                              # Where to place the World Bank dataset (see Data/README.md)
├── Docs/                             # Published manuscript (PDF), abstract, DOI, and links
├── Figures/                          # Generated figures (EDA, forecasts, comparisons)
└── Results/                          # Model comparison metrics (CSV)
```

## Results

Test-set performance across all nine models (see `Results/model_comparison.csv`):

| Rank | Model | RMSE | MAE | MAPE (%) | Category |
|------|-------|------|-----|----------|----------|
| 1 | XGB-LSTM | 0.2179 | 0.1702 | 0.98 | Hybrid |
| 2 | ARIMA | 0.3685 | 0.2807 | 1.56 | Statistical |
| 3 | GRU | 0.4603 | 0.3224 | 1.77 | Deep Learning |
| 4 | RNN | 0.4687 | 0.3238 | 1.78 | Deep Learning |
| 5 | Transformer | 0.6199 | 0.3847 | 2.10 | Deep Learning |
| 6 | CNN-LSTM | 0.6458 | 0.3919 | 2.13 | Hybrid |
| 7 | XGBoost | 0.7967 | 0.4544 | 2.46 | Machine Learning |
| 8 | LSTM | 1.2628 | 0.9858 | 5.43 | Deep Learning |
| 9 | ARIMA-LSTM | 1.8681 | 1.8449 | 10.49 | Hybrid |

> **Note:** The values above reflect this notebook's run. The published article reports a refined configuration (LSTM as champion, test window 2016–2020); rerun the notebook if you need to reproduce the exact published numbers.

## Figures

Generated plots are saved in `Figures/`: exploratory analysis (`fig_eda.png`), all-model forecasts (`fig_all_forecasts.png`), metric comparison (`fig_metric_comparison.png`), regional 20-year forecasts (`fig_regional_forecast.png`), structural break (`fig_structural_break.png`), phase RMSE (`fig_phase_rmse.png`), transition velocity index (`fig_tvi.png`), and beta convergence (`fig_beta_convergence.png`).

## Data

World Bank indicator **EG.FEC.RNEW.ZS** (Renewable energy consumption, % of total final energy consumption), 1960–2020, for 11 aggregate regions and income groups. Download the dataset from the [World Bank data portal](https://data.worldbank.org/indicator/EG.FEC.RNEW.ZS).

> **Note:** The notebook loads the dataset from a local file path (`FILEPATH` variable near the top). Update that path to point to your downloaded World Bank CSV/XLS before running.

## Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the World Bank dataset and update FILEPATH in the notebook

# 3. Launch the notebook
jupyter notebook forecasting_benchmark_v2_1.ipynb
```

Deep learning training uses PyTorch with hardware acceleration via Apple Metal Performance Shaders (MPS) when available, falling back to CUDA or CPU.

## Citation

```bibtex
@article{biswas2026renewable,
  title   = {Global Renewable Energy Consumption Forecasting: A Comparative Benchmarking Study of Statistical, Machine Learning, and Deep Learning Models},
  author  = {Biswas, Shaon and Irshad, Asadullah and Roy, Paramita},
  journal = {Computer Engineering and Intelligent Systems},
  volume  = {17},
  number  = {1},
  pages   = {44--57},
  year    = {2026},
  doi     = {10.7176/CEIS/17-1-05}
}
```

## License

The article is published under a Creative Commons Attribution 3.0 License. Copyrights for the article are retained by the authors.
