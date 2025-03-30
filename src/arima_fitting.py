import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from data_loader import get_cpi_data
import warnings

# Suppress statsmodels warnings for cleaner output
warnings.filterwarnings("ignore")

def fit_optimal_arima(p=2, d=1, q=3, plot_results=True):
    """
    Fits an ARIMA model with specified parameters and analyzes the residuals.
    
    Args:
        p (int): AR order (autoregressive lag), default is 2.
        d (int): Differencing order, default is 1.
        q (int): MA order (moving average lag), default is 3.
        plot_results (bool): Whether to generate diagnostic plots.
        
    Returns:
        tuple: Fitted model, DataFrame of diagnostics, and residuals
    """
    # Load data
    df = get_cpi_data()
    time_series_data = df['cpi']
    
    print(f"Fitting ARIMA({p},{d},{q}) model...")
    
    # Fit the ARIMA model
    model = ARIMA(time_series_data, order=(p, d, q))
    model_fit = model.fit()
    
    # Print model summary
    print(model_fit.summary())
    
    # Get residuals
    residuals = model_fit.resid
    
    # Create a diagnostic DataFrame
    diag_df = pd.DataFrame({
        'Mean': [residuals.mean()],
        'Std Dev': [residuals.std()],
        'Skewness': [residuals.skew()],
        'Kurtosis': [residuals.kurtosis()]
    })
    
    # Ljung-Box Test at multiple lags
    lags = [5, 10, 15, 20]
    lb_results = acorr_ljungbox(residuals, lags=lags, return_df=True)
    
    print("\nLjung-Box Test Results:")
    print(lb_results)
    
    # Interpretation of Ljung-Box results
    for lag in lags:
        p_value = lb_results.loc[lag, 'lb_pvalue']
        if p_value < 0.05:
            print(f"At lag {lag}: Significant autocorrelation remains (p-value: {p_value:.4f})")
        else:
            print(f"At lag {lag}: No significant autocorrelation (p-value: {p_value:.4f})")
    
    if plot_results:
        # Create diagnostic plots
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Original Series vs Fitted
        plt.subplot(2, 2, 1)
        plt.plot(time_series_data, label='Original')
        plt.plot(model_fit.fittedvalues, color='red', label='Fitted')
        plt.title('Original Series vs Fitted Values')
        plt.legend()
        
        # Plot 2: Residuals Time Series
        plt.subplot(2, 2, 2)
        plt.plot(residuals)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Residuals')
        
        # Plot 3: Residual Distribution
        plt.subplot(2, 2, 3)
        sns.histplot(residuals, kde=True, bins=25)
        plt.title('Residuals Distribution')
        
        # Plot 4: ACF of Residuals
        ax4 = plt.subplot(2, 2, 4)
        plot_acf(residuals, lags=20, ax=ax4)
        plt.title('ACF of Residuals')
        
        plt.tight_layout()
        plt.savefig("../results/graphs/arima_diagnostics.png")
        plt.show()
        
        # QQ Plot for residuals
        plt.figure(figsize=(8, 6))
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('QQ Plot of Residuals')
        plt.savefig("../results/graphs/residuals_qqplot.png")
        plt.show()
    
    return model_fit, diag_df, residuals

def forecast_future(model_fit, steps=12):
    """
    Generates forecasts for future periods using the fitted ARIMA model.
    
    Args:
        model_fit: Fitted ARIMA model
        steps (int): Number of steps to forecast
        
    Returns:
        pd.Series: Forecasted values with confidence intervals
    """
    # Get forecast
    forecast_result = model_fit.get_forecast(steps=steps)
    forecast_mean = forecast_result.predicted_mean
    forecast_ci = forecast_result.conf_int()
    
    # Plot forecast
    plt.figure(figsize=(12, 6))
    
    # Plot history
    df = get_cpi_data()
    plt.plot(df['cpi'], label='Historical CPI')
    
    # Plot forecast
    forecast_index = pd.date_range(
        start=df.index[-1] + pd.DateOffset(months=1),
        periods=steps,
        freq='MS'
    )
    forecast_mean.index = forecast_index
    forecast_ci.index = forecast_index
    
    plt.plot(forecast_mean, color='red', label='Forecast')
    plt.fill_between(
        forecast_ci.index,
        forecast_ci.iloc[:, 0],
        forecast_ci.iloc[:, 1],
        color='pink', alpha=0.3
    )
    
    plt.title(f'CPI Forecast for Next {steps} Months')
    plt.legend()
    plt.savefig("../results/graphs/cpi_forecast.png")
    plt.show()
    
    # Create forecast DataFrame with confidence intervals
    forecast_df = pd.DataFrame({
        'forecast': forecast_mean,
        'lower_ci': forecast_ci.iloc[:, 0],
        'upper_ci': forecast_ci.iloc[:, 1]
    })
    
    return forecast_df

if __name__ == "__main__":
    # Fit the optimal ARIMA model with p=2, d=1, q=3
    model_fit, diagnostics, residuals = fit_optimal_arima(p=2, d=1, q=3)
    
    print("\nResidual Diagnostics:")
    print(diagnostics)
    
    # Generate forecasts for the next 12 months
    print("\nGenerating forecasts...")
    forecast = forecast_future(model_fit, steps=12)
    print("\nForecasts for the next 12 months:")
    print(forecast)