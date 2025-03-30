import matplotlib.pyplot as plt
import statsmodels.api as sm
from data_loader import get_cpi_data
from stationarity import make_stationary

def plot_acf_pacf(df=None, column="cpi_diff"):
    """
    Plots the ACF and PACF for the specified column.
    
    Args:
        df: DataFrame containing the data. If None, uses stationary data from make_stationary().
        column: Column name to plot (default: "cpi_diff")
        
    Returns:
        The input DataFrame
    """
    if df is None:
        df = make_stationary()
    
    # Check if the column exists in the DataFrame
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame. Available columns: {df.columns.tolist()}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sm.graphics.tsa.plot_acf(df[column].dropna(), ax=axes[0], lags=20)
    axes[0].set_title("Autocorrelation (ACF)")
    
    sm.graphics.tsa.plot_pacf(df[column].dropna(), ax=axes[1], lags=20)
    axes[1].set_title("Partial Autocorrelation (PACF)")
    
    plt.savefig("../results/graphs/acf_pacf.png")
    plt.show()
    
    return df

if __name__ == "__main__":
    # Get stationary data and plot ACF/PACF
    print(get_cpi_data())
    stationary_data = make_stationary()
    plot_acf_pacf(stationary_data)