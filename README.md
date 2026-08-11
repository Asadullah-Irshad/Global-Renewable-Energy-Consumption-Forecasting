# Global Renewable Energy Consumption Forecasting

### A Comparative Benchmarking Study of Statistical, Machine Learning, and Deep Learning Models

> Benchmarking ARIMA, XGBoost, LSTM & Transformer for global renewable energy forecasting (World Bank, 1960–2020). Code for our CEIS 2026 paper.

**Tech stack:**
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-8CAAE6)
![pmdarima](https://img.shields.io/badge/pmdarima-ARIMA-1f77b4)
![XGBoost](https://img.shields.io/badge/XGBoost-EC4E20?logo=xgboost&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=python&logoColor=white)
![seaborn](https://img.shields.io/badge/seaborn-4B8BBE)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

Reference code (Jupyter notebook) for the published paper:

> **Biswas, S., Irshad, A., & Roy, P. (2026).** Global Renewable Energy Consumption Forecasting: A Comparative Benchmarking Study of Statistical, Machine Learning, and Deep Learning Models. *Computer Engineering and Intelligent Systems*, **17**(1), 44–57.
>
> **DOI:** [10.7176/CEIS/17-1-05](https://doi.org/10.7176/CEIS/17-1-05) · **Article:** [iiste.org](https://iiste.org/Journals/index.php/CEIS/article/view/63772) · published PDF: [`Docs/manuscript.pdf`](Docs/manuscript.pdf)
> Published March 28, 2026 · ISSN (Online) 2222-2863

## Abstract

Accurate forecasting of renewable energy consumption is essential for energy policy planning, infrastructure investment, and monitoring the global energy transition. This study presents a rigorous comparative benchmarking of four forecasting approaches — **ARIMA, XGBoost, LSTM, and Transformer** — applied to annual renewable energy consumption data from the World Bank (`EG.FEC.RNEW.ZS`, 1960–2020) across 11 aggregate regions and income groups. Each model undergoes automated hyperparameter optimisation, and predictive accuracy is evaluated on a held-out test period (2016–2020) using RMSE. Deep learning models outperform classical baselines: **LSTM achieves the best test RMSE (0.7286)**, followed by Transformer (0.8938), ARIMA (1.2294), and XGBoost (1.2518). The champion LSTM model is retrained per region to generate 20-year forecasts (2021–2040).

## What the study shows

**The World renewable share is non-stationary, with a sharp late-period acceleration.** It sits near 16.7% for decades, dips through the 2000s, then climbs steeply to 19.7% by 2020.

<p align="center">
  <img src="Figures/figure1_global_renewable_energy_trend.png" alt="Global renewable energy consumption trend, 1960–2020" width="85%">
</p>

*Figure 1 — Global renewable energy consumption trend (World aggregate, 1960–2020), % of total final energy.*

**A seasonal-trend decomposition** separates the long-run trend from residual structure in the World series.

<p align="center">
  <img src="Figures/figure2_seasonal_decomposition.png" alt="Additive seasonal decomposition of the World series" width="80%">
</p>

*Figure 2 — Additive decomposition of the World aggregate renewable-energy series.*

**The autocorrelation structure** (ACF / PACF) motivates the ARIMA specification and confirms strong persistence.

<p align="center">
  <img src="Figures/figure3_acf_pacf.png" alt="ACF and PACF of the World series" width="85%">
</p>

*Figure 3 — Autocorrelation (ACF) and partial autocorrelation (PACF) of the World series.*

**On the held-out test window,** the models are compared directly against the actual values.

<p align="center">
  <img src="Figures/figure4_forecasting_benchmarks.png" alt="Model predictions versus actual values on the test window" width="80%">
</p>

*Figure 4 — Forecasting benchmarks: model predictions vs. actual values for the World aggregate test period.*

**The champion LSTM is then retrained per region** to project the next two decades, revealing divergent regional transition trajectories.

<p align="center">
  <img src="Figures/figure5_lstm_20year_forecasts.png" alt="LSTM 20-year forecasts for all 11 regions" width="90%">
</p>

*Figure 5 — LSTM champion 20-year forecasts (2021–2040) for all 11 regions and income groups.*

## Results

**Table 3 — Test RMSE across the four benchmarked models** (World aggregate, test period 2016–2020):

| Model | Category | Test RMSE | Rank |
|---|---|---:|:--:|
| **LSTM** | Deep Learning | **0.7286** | 1 ★ |
| Transformer | Deep Learning | 0.8938 | 2 |
| ARIMA | Statistical | 1.2294 | 3 |
| XGBoost | Machine Learning | 1.2518 | 4 |

**Table 5 — Year-by-year actual vs. predicted** (test period; LSTM/Transformer point predictions available via the trained PyTorch models):

| Year | Actual (%) | ARIMA | XGBoost |
|---|---:|---:|---:|
| 2016 | 17.6281 | 17.4035 | 17.4038 |
| 2017 | 17.8542 | 17.4035 | 17.8762 |
| 2018 | 18.1052 | 17.4035 | 17.8002 |
| 2019 | 18.5752 | 17.4035 | 17.3651 |
| 2020 | 19.7356 | 17.4035 | 17.2401 |
| **RMSE** | — | **1.2294** | **1.2518** |

**Table 2 — Renewable energy consumption by region, 1960 vs. 2020** (% of total final energy):

| Region / Income Group | 1960 (%) | 2020 (%) | Trend |
|---|---:|---:|---|
| East Asia & Pacific | 26.42 | 14.81 | Declining |
| Europe & Central Asia | 5.72 | 15.11 | Strongly increasing |
| High income | 5.97 | 12.78 | Increasing |
| Latin America & Caribbean | 32.57 | 34.20 | Broadly stable / rising |
| Low income | 51.36 | 69.20 | Strongly increasing |
| Lower middle income | 55.50 | 41.64 | Declining |
| North America | 6.20 | 12.46 | Increasing |
| South Asia | 55.64 | 36.89 | Declining |
| Sub-Saharan Africa | 70.72 | 70.27 | Broadly stable |
| Upper middle income | 25.87 | 16.59 | Declining |
| World | 16.68 | 19.74 | Increasing (non-linear) |

The remaining published tables — **Table 1** (dataset summary) and **Table 4** (hyperparameter search spaces and optimal configurations) — are in [`Tables/`](Tables/) as CSVs alongside the three above.

> **Note on reproduction.** The published tables above are the authoritative results. The notebook also runs an *extended* nine-model benchmark (adding RNN, GRU, CNN-LSTM, ARIMA-LSTM, XGB-LSTM) for exploration; those extra rows are not part of the published paper.

## Repository structure

```
Global-Renewable-Energy-Consumption-Forecasting/
├── forecasting_benchmark_v2_1.ipynb   # full analysis notebook
├── Src/          # plain-Python (.py) version of the notebook
├── Figures/      # the 5 published manuscript figures
├── Tables/       # the 5 published manuscript tables (CSV)
├── Results/      # model-comparison metrics (CSV)
├── Data/         # where to place the World Bank dataset (see Data/README.md)
├── Docs/         # published manuscript (PDF), abstract, DOI, links
├── requirements.txt · README.md · LICENSE · CITATION.cff
```

## Data

World Bank indicator **EG.FEC.RNEW.ZS** (Renewable energy consumption, % of total final energy consumption), 1960–2020, for 11 aggregate regions and income groups. Download from the [World Bank data portal](https://data.worldbank.org/indicator/EG.FEC.RNEW.ZS) and place it in `Data/` (see `Data/README.md`).

> The notebook loads the dataset from the `FILEPATH` variable near the top — update it to point to your downloaded World Bank file before running.

## Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the World Bank dataset and update FILEPATH in the notebook

# 3. Launch the notebook
jupyter notebook forecasting_benchmark_v2_1.ipynb
```

Deep-learning training uses PyTorch with hardware acceleration via Apple Metal Performance Shaders (MPS) when available, falling back to CUDA or CPU.

## Author contributions

Roles follow the [CRediT taxonomy](https://credit.niso.org/):

- **Asadullah Irshad** — conceptualisation, methodology, software, validation, formal analysis, investigation, data curation, visualisation, resources (deep-learning compute), writing (review and editing).
- **Shaon Biswas** — conceptualisation, validation, writing (original draft), writing (review and editing), project administration.
- **Paramita Roy** — conceptualisation, validation, writing (original draft), writing (review and editing), project administration.

All authors read and approved the final manuscript.

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

`CITATION.cff` carries the machine-readable form — GitHub's **Cite this repository** button reads it directly.

## License

The article is published open access under a Creative Commons Attribution 3.0 License (CC BY 3.0); copyrights for the article are retained by the authors (Shaon Biswas, Asadullah Irshad, Paramita Roy).
