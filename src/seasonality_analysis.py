import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import seaborn as sns
from data_loader import get_cpi_data

def check_seasonality(freq='monthly'):
    """
    Check if the CPI data exhibits seasonality.
    
    Args:
        freq (str): Frequency of the data ('monthly' or 'quarterly')
        
    Returns:
        tuple: Seasonal decomposition result and boolean indicating if seasonality exists
    """
    # Load data
    df = get_cpi_data()
    
    # Ensure the data is properly indexed
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Warning: Index is not a DatetimeIndex, trying to convert...")
        df.index = pd.to_datetime(df.index)
    
    # Determine the period based on frequency
    period = 12 #for monthly
    
    # Perform seasonal decomposition
    result = seasonal_decompose(df['cpi'], model='additive', period=period)
    
    # Plot the decomposition
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
    
    result.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    
    result.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    
    result.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    
    result.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    
    plt.tight_layout()
    plt.savefig("../results/graphs/seasonal_decomposition.png")
    plt.show()
    
    # Check for significant seasonality
    seasonal_strength = np.abs(result.seasonal).max() / np.abs(result.trend).std()
    has_seasonality = seasonal_strength > 0.1  # Arbitrary threshold
    
    print(f"Seasonal strength: {seasonal_strength:.4f}")
    
    if has_seasonality:
        print("The data exhibits seasonality.")
        
        # Visualize the seasonal pattern
        seasonal_pattern = result.seasonal[:period]
        plt.figure(figsize=(10, 4))
        seasonal_pattern.plot(kind='bar')
        plt.title('Seasonal Pattern')
        plt.tight_layout()
        plt.savefig("../results/graphs/seasonal_pattern.png")
        plt.show()
    else:
        print("The data does not exhibit significant seasonality.")
    
    return result, has_seasonality

def fit_sarima_model(p=2, d=1, q=3, P=1, D=0, Q=1, s=12):
    """
    Fit a SARIMA model to the CPI data.
    
    Args:
        p, d, q: Non-seasonal ARIMA orders
        P, D, Q: Seasonal ARIMA orders
        s: Seasonal period
        
    Returns:
        model_fit: Fitted SARIMA model
    """
    # Load data
    df = get_cpi_data()
    
    print(f"Fitting SARIMA({p},{d},{q})({P},{D},{Q}){s} model...")
    
    # Fit the SARIMA model
    model = SARIMAX(df['cpi'], order=(p, d, q), seasonal_order=(P, D, Q, s))
    model_fit = model.fit(disp=False)
    
    # Print model summary
    print(model_fit.summary())
    
    # Get residuals
    residuals = model_fit.resid
    
    # Ljung-Box Test
    lags = [5, 10, 15, 20]
    lb_results = acorr_ljungbox(residuals, lags=lags, return_df=True)
    
    print("\nLjung-Box Test Results:")
    print(lb_results)
    
    # Plot diagnostics
    plt.figure(figsize=(14, 12))
    
    # Plot 1: Original Series vs Fitted
    plt.subplot(3, 2, 1)
    plt.plot(df['cpi'], label='Original')
    plt.plot(model_fit.fittedvalues, color='red', label='Fitted')
    plt.title('Original Series vs Fitted Values')
    plt.legend()
    
    # Plot 2: Residuals Time Series
    plt.subplot(3, 2, 2)
    plt.plot(residuals)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Residuals')
    
    # Plot 3: Residual Distribution
    plt.subplot(3, 2, 3)
    sns.histplot(residuals, kde=True, bins=25)
    plt.title('Residuals Distribution')
    
    # Plot 4: ACF of Residuals
    plt.subplot(3, 2, 4)
    plot_acf(residuals, lags=20)
    plt.title('ACF of Residuals')
    
    # Plot 5: PACF of Residuals
    plt.subplot(3, 2, 5)
    plot_pacf(residuals, lags=20)
    plt.title('PACF of Residuals')
    
    # Plot 6: QQ plot
    plt.subplot(3, 2, 6)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('QQ Plot of Residuals')
    
    plt.tight_layout()
    plt.savefig("../results/graphs/sarima_diagnostics.png")
    plt.show()
    
    return model_fit

def forecast_with_sarima(model_fit, steps=12):
    """
    Generate forecasts using the fitted SARIMA model.
    
    Args:
        model_fit: Fitted SARIMA model
        steps: Number of steps to forecast
        
    Returns:
        forecast_df: DataFrame containing forecasts and confidence intervals
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
    
    plt.title(f'CPI Forecast for Next {steps} Months (with Seasonality)')
    plt.legend()
    plt.savefig("../results/graphs/seasonal_cpi_forecast.png")
    plt.show()
    
    # Create forecast DataFrame with confidence intervals
    forecast_df = pd.DataFrame({
        'forecast': forecast_mean,
        'lower_ci': forecast_ci.iloc[:, 0],
        'upper_ci': forecast_ci.iloc[:, 1]
    })
    
    return forecast_df

def decompose_and_forecast(steps=12):
    """
    Decompose the time series, model each component separately, and create combined forecasts.
    
    Args:
        steps: Number of steps to forecast
        
    Returns:
        combined_forecast: DataFrame with combined forecasts
    """
    # Load data
    df = get_cpi_data()
    
    # Decompose the series
    decomposition = seasonal_decompose(df['cpi'], model='additive', period=12)
    
    # Extract components
    trend = decomposition.trend.dropna()
    seasonal = decomposition.seasonal.dropna()
    residual = decomposition.resid.dropna()
    
    # Align indices
    common_idx = trend.index.intersection(seasonal.index).intersection(residual.index)
    trend = trend.loc[common_idx]
    seasonal = seasonal.loc[common_idx]
    residual = residual.loc[common_idx]
    
    # Model trend component with ARIMA
    print("Modeling trend component...")
    from statsmodels.tsa.arima.model import ARIMA
    trend_model = ARIMA(trend, order=(2, 1, 2))
    trend_fit = trend_model.fit()
    
    # Model residual component with ARIMA
    print("Modeling residual component...")
    resid_model = ARIMA(residual, order=(1, 0, 1))
    resid_fit = resid_model.fit()
    
    # Generate forecasts for trend
    trend_forecast = trend_fit.forecast(steps=steps)
    
    # Generate forecasts for residuals
    resid_forecast = resid_fit.forecast(steps=steps)
    
    # Create future dates
    future_dates = pd.date_range(start=common_idx[-1] + pd.DateOffset(months=1), 
                                periods=steps, freq='MS')
    
    # Create seasonal pattern for future dates
    future_seasonal = pd.Series(
        [seasonal[seasonal.index.month == month].mean() for month in future_dates.month],
        index=future_dates
    )
    
    # Combine forecasts
    combined_forecast = pd.DataFrame({
        'trend_forecast': trend_forecast,
        'seasonal_forecast': future_seasonal,
        'residual_forecast': resid_forecast,
        'combined_forecast': trend_forecast + future_seasonal + resid_forecast
    })
    
    # Plot combined forecast
    plt.figure(figsize=(14, 8))
    
    # Plot history
    plt.plot(df['cpi'], label='Historical CPI')
    
    # Plot components and combined forecast
    plt.plot(combined_forecast.index, combined_forecast['trend_forecast'], 
             linestyle='--', label='Trend Forecast')
    plt.plot(combined_forecast.index, combined_forecast['seasonal_forecast'], 
             linestyle='--', label='Seasonal Component')
    plt.plot(combined_forecast.index, combined_forecast['residual_forecast'], 
             linestyle='--', label='Residual Forecast')
    plt.plot(combined_forecast.index, combined_forecast['combined_forecast'], 
             color='red', linewidth=2, label='Combined Forecast')
    
    plt.title('Component-wise Forecast for CPI')
    plt.legend()
    plt.savefig("../results/graphs/component_forecast.png")
    plt.show()
    
    return combined_forecast

def compare_models(steps=12):
    """
    Compare ARIMA, SARIMA, and decomposition-based forecasts.
    
    Args:
        steps: Number of steps to forecast
        
    Returns:
        comparison_df: DataFrame with forecasts from all models
    """
    # Load data
    df = get_cpi_data()
    
    # Check seasonality
    decomposition, has_seasonality = check_seasonality()
    
    # If there's seasonality, run all models
    if has_seasonality:
        print("\nRunning multiple forecasting models for comparison...")
        
        # 1. ARIMA model
        from arima_fitting import fit_optimal_arima, forecast_future
        arima_model, _, _ = fit_optimal_arima(p=2, d=1, q=3, plot_results=False)
        arima_forecast = forecast_future(arima_model, steps=steps)
        
        # 2. SARIMA model
        sarima_model = fit_sarima_model(p=2, d=1, q=3, P=1, D=0, Q=1, s=12)
        sarima_forecast = forecast_with_sarima(sarima_model, steps=steps)
        
        # 3. Decomposition-based forecast
        component_forecast = decompose_and_forecast(steps=steps)
        
        # Create comparison DataFrame
        future_dates = pd.date_range(
            start=df.index[-1] + pd.DateOffset(months=1),
            periods=steps, 
            freq='MS'
        )
        
        comparison_df = pd.DataFrame({
            'ARIMA': arima_forecast['forecast'],
            'SARIMA': sarima_forecast['forecast'],
            'Component': component_forecast['combined_forecast']
        })
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        plt.plot(df['cpi'].tail(24), label='Historical CPI')
        plt.plot(comparison_df.index, comparison_df['ARIMA'], label='ARIMA Forecast')
        plt.plot(comparison_df.index, comparison_df['SARIMA'], label='SARIMA Forecast')
        plt.plot(comparison_df.index, comparison_df['Component'], label='Component Forecast')
        plt.title('Forecast Comparison')
        plt.legend()
        plt.savefig("../results/graphs/forecast_comparison.png")
        plt.show()
        
        return comparison_df
    else:
        print("No significant seasonality detected. ARIMA model should be sufficient.")
        # Just run ARIMA if no seasonality
        from arima_fitting import fit_optimal_arima, forecast_future
        arima_model, _, _ = fit_optimal_arima(p=2, d=1, q=3)
        arima_forecast = forecast_future(arima_model, steps=steps)
        return arima_forecast

if __name__ == "__main__":
    # First check if there's seasonality
    print("Step 1: Checking for seasonality in CPI data...")
    decomp_result, has_seasonality = check_seasonality()
    
    if has_seasonality:
        # Compare different forecasting approaches
        print("\nStep 2: Comparing different forecasting models...")
        comparison = compare_models(steps=12)
        print("\nForecast comparison results:")
        print(comparison)
    else:
        # If no seasonality, just use ARIMA
        print("\nNo significant seasonality found. Using standard ARIMA model...")
        from arima_fitting import fit_optimal_arima, forecast_future
        model, _, _ = fit_optimal_arima(p=2, d=1, q=3)
        forecast = forecast_future(model, steps=12)
        print("\nARIMA forecasts:")
        print(forecast)