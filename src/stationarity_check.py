from statsmodels.tsa.stattools import adfuller
from data_loader import get_cpi_data

def check_stationarity(df=None, column="cpi"):
    """
    Performs the Augmented Dickey-Fuller test to check for stationarity.
    
    Args:
        df: DataFrame containing CPI data. If None, loads from data_loader.
        column: Column name to check for stationarity
        
    Returns:
        The input DataFrame and a boolean indicating if the series is stationary
    """
    if df is None:
        df = get_cpi_data()
        
    result = adfuller(df[column])
    
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])
    
    is_stationary = result[1] <= 0.05
    
    if not is_stationary:
        print("The series is non-stationary.")
    else:
        print("The series is stationary.")
    
    return df, is_stationary

if __name__ == "__main__":
    check_stationarity()