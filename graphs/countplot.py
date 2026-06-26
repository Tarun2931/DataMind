# MAINLY FOR CATEGORICAL DATA 
import matplotlib.pyplot as plt
import seaborn as sns
def create_countplot(df, column):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.countplot(x=df[column],ax=ax)
    ax.set_title(f"Countplot of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    return fig