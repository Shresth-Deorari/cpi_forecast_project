import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from data_loader import cpi_data

def check_stationarity(df, column="cpi"):
    """Performs the Augmented Dickey-Fuller test to check for stationarity."""
    result = adfuller(df[column])

    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])

    if result[1] > 0.05:
        print("The series is non-stationary.")
    else:
        print("The series is stationary.")

if __name__ == "__main__":
    check_stationarity(cpi_data, column="cpi") 
