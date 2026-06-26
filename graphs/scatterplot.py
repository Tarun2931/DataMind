#COLUMNS KE RP KE LIYE
import matplotlib.pyplot as plt
def create_scatter_plot(df, x, y):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(df[x],df[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}")
    return fig