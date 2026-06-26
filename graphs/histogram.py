#DISTRIBUTION OF DATA KE LIYE 
import matplotlib.pyplot as plt
def create_histogram(df, column):
    plt.figure(figsize=(8,5))
    plt.hist(df[column],bins=20)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {column}")
    return plt