#OUTLIERS SAMNJHNE KE LIYE 
import matplotlib.pyplot as plt
def create_boxplot(df, column):
    plt.figure(figsize=(8,5))
    plt.boxplot(df[column])
    plt.ylabel(column)
    plt.title(f"Boxplot of {column}")
    return plt