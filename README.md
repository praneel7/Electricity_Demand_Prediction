# ⚡ Forecasting Household Electricity Demand Using Machine Learning and Time Series Models

##  Overview

With growing emphasis on smart energy management, accurate prediction of household electricity demand is vital for planning, optimization, and sustainability efforts. This project uses daily electricity consumption data from a single household (sourced from the UCI dataset) and explores multiple forecasting strategies to model and predict **Global Active Power** consumption.

We apply a structured approach combining:
- Multivariate feature engineering
- Outlier detection using Facebook Prophet
- Lag feature construction to reduce data leakage
- Modeling via **SARIMAX**, **Support Vector Regression (SVR)**, and **LSTM**

The entire workflow is rigorously validated through **nested cross-validation**, with final evaluation performed on a **holdout dataset** (last 200 observations).

## Goals

- Forecast future household electricity demand with interpretable and robust models
- Compare the performance of statistical and ML methods in a multivariate time series setting
- Apply anomaly detection and lag-aware preprocessing to avoid overfitting
- Use realistic error metrics: MAE, RMSE, and MAPE

## Project Structure

```
.
├── New_Data_CleanAndProcessing.ipynb   # Cleaning, feature engineering, Prophet-based anomaly detection
├── Final_Report.ipynb                  # Nested CV, hyperparameter search, model training, evaluation
├── README.md                           # Project overview and full methodology (this file)
```

## Dataset Description

**Source**: [UCI Household Electric Power Consumption Dataset](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

**Granularity**: Daily

**Total Observations**: ~1,400 days (excluding missing dates)

### Key Variables:
| Feature | Description |
|--------|-------------|
| Global_active_power | Total daily energy used (target) |
| Sub_metering_1/2/3 | Appliance group-wise consumption |
| Voltage, Global_intensity | Raw electrical parameters |
| Weather: Temp, Precipitation, Sunshine | External regressors from local climate data |
| Derived: non_submetered | Estimated unaccounted consumption |
| Calendar: DayOfWeek, IsWeekend, Month | Seasonal and behavioral indicators |

## Preprocessing Pipeline

### Cleaning & Feature Construction
- Removed missing and non-numeric rows
- Computed `non_submetered` energy as a residual
- Merged external daily weather data (temperature, rain, sunshine)

### Outlier Detection with Prophet
- Trained a univariate Prophet model on the target
- Identified significant residual spikes (anomalies)
- Imputed these with fitted values for stability

### Lagging to Avoid Leakage
- Submetering values (e.g., kitchen/laundry/climate) are highly correlated with the target.
- Without lag, models overfit by "learning" the target trivially.
- We introduced **1-day lags** on all submetering features to prevent this.

##  Modeling Methodology

### Models Compared
| Model     | Notes |
|-----------|-------|
| **SARIMAX** | Interpretable; handles seasonality + exogenous regressors |
| **SVR**     | Captures non-linear relationships; top performer |
| **LSTM**    | Neural network for sequences; moderate performance |

###  Nested Cross-Validation

- **Outer loop**: Time-based splits for model evaluation (train-test on past→future)
- **Inner loop**: Random search over hyperparameters
- **Metrics**: MAE, RMSE, MAPE; SARIMAX additionally uses AIC and Durbin-Watson

###  Hyperparameter Tuning
- SVR: `C`, `epsilon`, `kernel`
- SARIMAX: `(p,d,q)(P,D,Q,s)`
- LSTM: hidden size, layers, sequence length

## Final Model Evaluation on Holdout Set

After cross-validation:
- The best config for each model was re-trained on full training data
- Evaluation was done on a **holdout set of the last 200 days**, excluded from CV

| Model     | MAE | RMSE | MAPE |
|-----------|-----|------|------|
| **SVR**   | ~260 | — | — |
| SARIMAX   | ~305 | — | — |
| LSTM      | ~320+ | — | — |

**Visualizations** in the notebook compare predicted vs actuals for each model.

## Insights

### SVR performed best
- With lagged features, SVR consistently delivered the lowest MAE.
- Effective at capturing subtle non-linear patterns.

### SARIMAX was interpretable and stable
- Strong when seasonal order was tuned well
- Captured broader trends and weekly/monthly seasonality

### LSTM showed potential but lagged
- Moderate performance, possibly due to:
  - Lack of strong repeating sequences
  - Small dataset size for deep learning

### Importance of Lagging
- Without lag, models "memorized" the target from submetering
- Lagging introduced temporal realism and prevented leakage

## Authors

- **Praneel Panchigar**  
  M.S. Data Analytics & Statistics, Washington University in St. Louis  
- **Gino Kler**
  M.S. Data Analytics & Statistics, Washington University in St. Louis 

## References

1. Taylor & Letham (2018), *Forecasting at Scale* (Prophet)
2. Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*
3. Fan et al. (2017), *Applied Energy* — LSTM for cooling demand
4. Weron (2014), *Electricity Price Forecasting Review*
5. UCI ML Repository: Individual Household Electric Power Consumption
