# CPI Time Series Analysis and Forecasting

## PROJECT OVERVIEW
Welcome to the CPI Time Series Analysis and Forecasting project! This repository provides a modular framework for analyzing Consumer Price Index (CPI) data using time series methods. It includes data loading, stationarity checks, ARIMA/SARIMA modeling, seasonality detection, and forecasting.

> Current status: Core functionalities are implemented with a focus on modularity.

## PROJECT STRUCTURE
The repository is organized into the following files, each serving a specific purpose:

- `data_loader.py`: Loads and prepares the CPI dataset.
- `check_stationarity.py`: Tests for stationarity using the Augmented Dickey-Fuller (ADF) test.
- `make_stationary.py`: Applies differencing to achieve stationarity.
- `plot_acf_pacf.py`: Generates ACF and PACF plots for parameter selection.
- `calculate_aic_bic.py`: Evaluates differencing orders using AIC and BIC.
- `arima_fitting.py`: Fits ARIMA models, performs diagnostics, and generates forecasts.
- `plot_cpi_trend.py`: Plots the raw CPI trend.
- `seasonality_analysis.py`: Detects seasonality, fits models, and compares forecasts.

## INSTALLATION
To set up the project, ensure you have Python installed along with the necessary packages.

```
# Run the following command in your terminal:
pip install pandas numpy matplotlib seaborn statsmodels scipy
```

Dependencies:
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scipy

## USAGE
Below is a guide to using the project.

### 1. LOAD THE DATA
Use `data_loader.py` to load your CPI dataset. The dataset should have date and cpi columns.

```
# In data_loader.py:
# Define a function like load_cpi_data(filepath="../data/financial_market_cpi_india.csv", force_reload=False)
# Add your data loading and cleaning code here
```

### 2. CHECK STATIONARITY
Run `check_stationarity.py` to check if the CPI data is stationary.

```
# In check_stationarity.py:
# Define a function like check_stationarity(df=None, column="cpi")
# Insert your ADF test implementation here
```

### 3. MAKE IT STATIONARY
If the data is non-stationary, use `make_stationary.py` to apply differencing.

```
# In make_stationary.py:
# Define a function like make_stationary(df=None)
# Add your differencing logic here
```

### 4. PLOT ACF/PACF
Generate ACF and PACF plots with `plot_acf_pacf.py` to help select ARIMA parameters.

```
# In plot_acf_pacf.py:
# Define a function like plot_acf_pacf(df=None, column="cpi_diff")
# Insert your plotting code for ACF and PACF here
```

### 5. CALCULATE AIC/BIC
Use `calculate_aic_bic.py` to evaluate different differencing orders.

```
# In calculate_aic_bic.py:
# Define a function like calculate_aic_bic(p=2, q=3, d_range=(0, 2))
# Add your AIC/BIC calculation code here
```

### 6. FIT ARIMA MODEL
Fit an ARIMA model and generate forecasts with `arima_fitting.py`.

```
# In arima_fitting.py:
# Define functions like fit_optimal_arima(p=2, d=1, q=3, plot_results=True)
# Add your ARIMA model fitting code here

# Define forecast_future(model_fit, steps=12)
# Add your forecasting code here
```

### 7. PLOT CPI TREND
Visualize the raw CPI trend using `plot_cpi_trend.py`.

```
# In plot_cpi_trend.py:
# Define a function like plot_cpi_trend()
# Insert your CPI trend plotting code here
```

### 8. SEASONALITY ANALYSIS
Detect seasonality and compare models with `seasonality_analysis.py`.

```
# In seasonality_analysis.py:
# Define functions like check_seasonality(freq='monthly')
# Add your seasonality detection code here

# Define compare_models(steps=12)
# Add your model comparison code here
```

### RUN THE FULL ANALYSIS
To execute the entire analysis pipeline, run the following command:

```
# In your terminal:
python seasonality_analysis.py
# This will load the data, check for seasonality, fit models, and generate forecasts
```

## FUTURE PLANS
The following enhancements are planned to extend the project's capabilities:

- **Automated Model Selection Pipeline**: Add a pipeline to automatically select the best model somehow (details TBD—maybe based on AIC, BIC, or cross-validation). This will streamline the process of finding the optimal model configuration.
- **Frontend Interface with React**: Build a React-based frontend to let users interact with the analysis, visualize results, and tweak parameters through a slick UI.
- **Backend with Django**: Set up a Django backend to handle data processing, model fitting, and serve results to the frontend, making the project a full-stack app.

These upgrades will make the project more powerful and user-friendly—stay tuned!

## CONTRIBUTING
Contributions are welcome! To contribute:

```
# 1. Fork the repository
# 2. Create a new branch for your feature or bugfix
# 3. Make your changes and commit them
# 4. Submit a pull request with a clear description of your changes
```

## LICENSE
This project is licensed under the MIT License. Feel free to use and modify it, but please provide attribution.