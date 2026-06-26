#RELATION BETWEEN NUMERICAL COLUMN 
import matplotlib.pyplot as plt
import seaborn as sns
def create_heatmap(df):
    correlation_matrix = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(correlation_matrix,annot=True,cmap="coolwarm",ax=ax)
    ax.set_title("Heatmap")
    return fig