from statsmodels.tsa.stattools import adfuller
from data_loader import get_cpi_data

def make_stationary(df=None):
    """
    Makes the CPI data stationary through differencing.
    
    Args:
        df: DataFrame containing CPI data. If None, loads from data_loader.
        
    Returns:
        DataFrame with the stationary series added as columns
    """
    if df is None:
        df = get_cpi_data()
    
    # Create a copy to avoid modifying the original
    stationary_df = df.copy()
    
    # First differencing
    stationary_df["cpi_diff"] = stationary_df["cpi"].diff()
    result = adfuller(stationary_df["cpi_diff"].dropna())
    print("ADF Statistic (After Differencing):", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])
    
    is_stationary = result[1] <= 0.05
    
    # Second differencing if needed
    if not is_stationary:
        print("Still Non-Stationary, Applying Second Differencing...")
        stationary_df["cpi_diff2"] = stationary_df["cpi_diff"].diff()
        result = adfuller(stationary_df["cpi_diff2"].dropna())
        print("ADF Statistic (Second Differencing):", result[0])
        print("p-value:", result[1])
        print("Critical Values:", result[4])
        is_stationary = result[1] <= 0.05
    
    if is_stationary:
        print("Data has become stationary")
    else:
        print("Warning: Data may still not be stationary")
    
    return stationary_df

if __name__ == "__main__":
    stationary_data = make_stationary()
    print("Available columns:", stationary_data.columns.tolist())