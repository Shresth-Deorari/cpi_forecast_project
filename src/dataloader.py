import pandas as pd

def load_cpi_data(filepath):
    """Loads the CPI dataset from a CSV file."""
    df = pd.read_csv(filepath)
    print(df.head())  # Check the first few rows
    print(df.info())  # Check data types and missing values
    return df

if __name__ == "__main__":
    df = load_cpi_data("../data/cpi_data.csv")