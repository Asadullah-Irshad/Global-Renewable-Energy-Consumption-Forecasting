# Paper

**Title:** Global Renewable Energy Consumption Forecasting: A Comparative Benchmarking Study of Statistical, Machine Learning, and Deep Learning Models

**Authors:** Shaon Biswas, Asadullah Irshad, Paramita Roy

**Journal:** Computer Engineering and Intelligent Systems (IISTE), Vol. 17, No. 1 (2026), pp. 44-57

**DOI:** [10.7176/CEIS/17-1-05](https://doi.org/10.7176/CEIS/17-1-05)

**Article page:** https://iiste.org/Journals/index.php/CEIS/article/view/63772

**Full-text PDF:** https://iiste.org/Journals/index.php/CEIS/article/download/63772/65924

**Local copy:** [`manuscript.pdf`](manuscript.pdf) (published version, 14 pages)

**Published:** March 28, 2026 - ISSN (Online) 2222-2863 - CC BY 3.0

## Abstract

Energy independence is a critical component of national sovereignty and economic security. As fossil fuel resources are geographically concentrated and global energy markets are strongly influenced by geopolitical dynamics, many countries are increasingly transitioning toward renewable energy sources. Renewable energy not only enhances energy security but also contributes to global climate objectives, including the United Nations Sustainable Development Goal 7 (SDG-7) and the International Energy Agency's Net-Zero Emissions scenario. In this context, accurate forecasting of renewable energy consumption is essential for energy policy planning, infrastructure investment, and monitoring the progress of the global energy transition.

This study presents a rigorous comparative benchmarking of four forecasting approaches - ARIMA, XGBoost, LSTM, and Transformer - applied to annual renewable energy consumption data sourced from the World Bank (EG.FEC.RNEW.ZS indicator, 1960-2020) across 11 aggregate regions and income groups. Each model undergoes automated hyperparameter optimisation, and predictive accuracy is evaluated on a held-out test period using RMSE. The World aggregate renewable energy share ranged from 16.54% (2007) to 19.74% (2020), with the Augmented Dickey-Fuller test confirming non-stationarity (ADF = 0.5240, p = 0.9856).

Results show that deep learning models outperform classical baselines: LSTM achieves the best test RMSE (0.7286), followed by Transformer (0.8938), ARIMA (1.2294), and XGBoost (1.2518). The champion LSTM model is subsequently retrained on each of the 11 regions and used to generate 20-year forecasts (2021-2040), revealing divergent regional energy transition trajectories. Hardware acceleration via Apple Metal Performance Shaders (MPS) on PyTorch was employed throughout deep learning training.

**Keywords:** Renewable energy forecasting; LSTM; Transformer; ARIMA; XGBoost; time series benchmarking; deep learning; World Bank; energy transition; PyTorch MPS
