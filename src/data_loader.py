import pandas as pd

def load_cpi_data(filepath="../data/financial_market_cpi_india.csv"):
    df = pd.read_csv(filepath)
    df.columns = ['date', 'cpi']
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df  # Return DataFrame

cpi_data = load_cpi_data()
