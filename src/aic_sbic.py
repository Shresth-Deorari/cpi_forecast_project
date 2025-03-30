import pandas as pd
import statsmodels.api as sm
from data_loader import get_cpi_data

def calculate_aic_bic(p=2, q=3, d_range=(0, 2)):
    """
    Calculates AIC and BIC for ARIMA models with different differencing orders (d).
    
    Args:
        p (int): AR order (from PACF), default is 2.
        q (int): MA order (from ACF), default is 3.
        d_range (tuple): Tuple specifying the range of d to test (inclusive), default is (0, 2).
        
    Returns:
        pd.DataFrame: DataFrame with columns 'd', 'AIC', and 'BIC' for each model.
    """
    # Load the original CPI data
    df = get_cpi_data()  # Uses the function from data_loader.py
    series = df["cpi"]
    
    # List to store results
    results = []
    
    # Iterate over the range of d values
    for d in range(d_range[0], d_range[1] + 1):
        try:
            # Define and fit the ARIMA model
            model = sm.tsa.ARIMA(series, order=(p, d, q))
            model_fit = model.fit()
            
            # Extract AIC and BIC
            aic = model_fit.aic
            bic = model_fit.bic
            
            # Append results
            results.append({"d": d, "AIC": aic, "BIC": bic})
            print(f"ARIMA({p},{d},{q}) - AIC: {aic:.2f}, BIC: {bic:.2f}")
        except Exception as e:
            print(f"Error fitting ARIMA({p},{d},{q}): {e}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def find_best_d(results_df):
    """
    Identifies the d value with the lowest AIC and BIC.
    
    Args:
        results_df (pd.DataFrame): DataFrame with 'd', 'AIC', and 'BIC' columns.
        
    Returns:
        dict: Dictionary with the best d based on AIC and BIC.
    """
    best_aic = results_df.loc[results_df["AIC"].idxmin()]
    best_bic = results_df.loc[results_df["BIC"].idxmin()]
    return {"best_d_aic": int(best_aic["d"]), "best_d_bic": int(best_bic["d"])}

if __name__ == "__main__":
    # Calculate AIC and BIC for d in range 0 to 2
    results = calculate_aic_bic(p=2, q=3, d_range=(0, 2))
    
    # Display results
    print("\nAIC and BIC for different d values:")
    print(results)
    
    # Find and display the best d
    best_d = find_best_d(results)
    print(f"\nBest d based on AIC: {best_d['best_d_aic']}")
    print(f"Best d based on BIC: {best_d['best_d_bic']}")