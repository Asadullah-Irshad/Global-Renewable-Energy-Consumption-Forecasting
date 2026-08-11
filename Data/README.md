# Data

This project uses the World Bank indicator **EG.FEC.RNEW.ZS** — *Renewable energy consumption (% of total final energy consumption)*.

## How to get the data

1. Download the dataset from the World Bank data portal:
   https://data.worldbank.org/indicator/EG.FEC.RNEW.ZS
2. Place the downloaded file in this `Data/` folder. The notebook expects:
   `API_EG.FEC.RNEW.ZS_DS2_en_csv_v2_3233.xls`
3. If your filename or location differs, update the `FILEPATH` variable near the top of `forecasting_benchmark_v2_1.ipynb`.

## Coverage used in the study

- Years: 1960–2020 (2021–2022 excluded due to incomplete reporting)
- 11 aggregate regions and income groups (East Asia & Pacific, Europe & Central Asia,
  High income, Latin America & Caribbean, Low income, Lower middle income, North America,
  South Asia, Sub-Saharan Africa, Upper middle income, World)

The raw World Bank file is not redistributed here; please download it directly from the source above.
