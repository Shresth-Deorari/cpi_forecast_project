import matplotlib.pyplot as plt
from data_loader import get_cpi_data

def plot_cpi_trend():
    """Plots the CPI trend over time and saves the figure."""
    cpi_data = get_cpi_data()
    
    plt.figure(figsize=(10, 5))
    plt.plot(cpi_data, label="CPI Index")
    plt.title("CPI Over Time")
    plt.xlabel("Year")
    plt.ylabel("CPI Value")
    plt.legend()
    plt.savefig("../results/graphs/cpi_trend.png")
    plt.show()
    
    return cpi_data  # Return the data for potential chaining

if __name__ == "__main__":
    plot_cpi_trend()