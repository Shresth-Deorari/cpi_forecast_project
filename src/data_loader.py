import pandas as pd

_cpi_data = None  # Private variable

def load_cpi_data(filepath="../data/financial_market_cpi_india.csv", force_reload=False):
    """
    Loads the CPI dataset and sets appropriate column names.
    
    Args:
        filepath: Path to the CSV file containing CPI data
        force_reload: Whether to reload the data even if it's already loaded
        
    Returns:
        DataFrame containing CPI data
    """
    global _cpi_data
    if _cpi_data is None or force_reload:
        df = pd.read_csv(filepath)
        df.columns = ['date', 'cpi']
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        _cpi_data = df  
    return _cpi_data  # Always return the current data

def get_cpi_data():
    """Returns the current CPI data."""
    global _cpi_data
    if _cpi_data is None:
        load_cpi_data()
    return _cpi_data.copy()  # Return a copy to prevent unintentional modifications

# Initialize data when module is imported
load_cpi_data()