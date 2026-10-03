\# 📈 Predictive Analytics Using Historical Data



Forecast future sales trends using \*\*regression, Prophet, and LSTM\*\* models with external features (holidays, promotions, weather).



\---



\## 🎯 Objective



Build a predictive model that:

\- Learns from historical time-series data

\- Handles trend, seasonality, and external factors

\- Forecasts future trends with confidence intervals

\- Is deployed as an interactive dashboard



\---



\## 📊 Dataset



\*\*Synthetic monthly sales data (2015–2024)\*\* with 4 features:



| Column | Description |

|--------|-------------|

| `ds` | Date (monthly) |

| `y` | Sales (target) |

| `promo` | Promotion active (0/1) — adds \~20 units |

| `weather` | Weather index — sinusoidal pattern |

| `is\_holiday` | Holiday season flag (Nov/Dec) |



The data includes a \*\*linear upward trend\*\*, \*\*yearly seasonality\*\*, \*\*holiday spikes\*\*, and \*\*random noise\*\*.



\---



\## 🛠️ Methods



| Step | Technique |

|------|-----------|

| 1. Data generation | Synthetic signal + external regressors |

| 2. Cleaning | Interpolation + IQR outlier clipping |

| 3. Linear Regression | Baseline model |

| 4. Prophet | Trend + seasonality + holiday + regressors |

| 5. LSTM | Multivariate sequence model (TensorFlow) |

| 6. Cross-validation | TimeSeriesSplit (5 folds) |

| 7. Dashboard | Streamlit + Plotly |



\---



\## 📈 Results



\### Model Performance (12-month holdout test)



| Model | MAE | RMSE | R² |

|-------|-----|------|----|

| Linear Regression | 6.21 | 7.35 | 0.930 |

| \*\*Prophet (+regressors)\*\* | \*\*5.21\*\* | \*\*6.84\*\* | \*\*0.971\*\* |

| LSTM (multivariate) | 7.85 | 9.62 | 0.912 |



\### Cross-Validation (TimeSeriesSplit, Mean across folds)



| Model | Mean MAE | Mean RMSE |

|-------|----------|-----------|

| Linear Regression | 21.47 | 25.80 |

| \*\*Prophet\*\* | \*\*9.44\*\* | \*\*11.60\*\* |



\*\*Prophet outperforms Linear Regression by 2.3× in cross-validation\*\* — its built-in decomposition of trend, seasonality, holidays, and external regressors handles complex patterns better.



\---



\## 🏆 Best Model: Prophet



\*\*Why Prophet wins:\*\*

\- ✅ Handles \*\*missing data\*\* and \*\*outliers\*\* naturally

\- ✅ Built-in \*\*holiday calendar\*\* (US)

\- ✅ Supports \*\*external regressors\*\* (promo, weather)

\- ✅ Decomposes into \*\*interpretable components\*\*

\- ✅ Provides \*\*uncertainty intervals\*\* for forecasts



\---



\## 🌐 Interactive Dashboard



The project includes a \*\*Streamlit dashboard\*\* where users can:

\- Adjust the \*\*forecast horizon\*\* (3–24 months)

\- Toggle \*\*promotions, weather, holidays\*\*

\- View \*\*confidence intervals\*\*

\- Download forecasts as CSV



\*\*To launch:\*\*

```bash

streamlit run step6\_streamlit\_app.py

