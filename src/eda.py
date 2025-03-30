import matplotlib.pyplot as plt
import seaborn as sns

def plot_cpi_trend(df, save_path="../results/graphs/cpi_trend.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(df["CPALTT01INM657N"], label="CPI", color="blue")
    plt.xlabel("Year")
    plt.ylabel("CPI Value")
    plt.title("CPI Trend Over Time")
    plt.legend()
    plt.savefig(save_path)
    # plt.show()

if __name__ == "__main__":
    from dataloader import load_cpi_data
    df = load_cpi_data("../data/cpi_data.csv")
    plot_cpi_trend(df)
