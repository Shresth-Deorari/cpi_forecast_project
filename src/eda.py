from data_loader import cpi_data
import matplotlib.pyplot as plt

def plot_cpi_trend():
    plt.figure(figsize=(10, 5))
    plt.plot(cpi_data, label="CPI Index")
    plt.title("CPI Over Time")
    plt.xlabel("Year")
    plt.ylabel("CPI Value")
    plt.legend()
    plt.savefig("../results/graphs/cpi_trend.png")
    plt.show()

if __name__ == "__main__":
    plot_cpi_trend()