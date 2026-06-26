import matplotlib.pyplot as plt
def create_residual_plot(actual, predicted):
    residuals = [a - p for a, p in zip(actual, predicted)]
    fig = plt.figure(figsize=(8,6))
    plt.scatter(predicted, residuals)
    plt.axhline(y=0,color="red",linestyle="--")
    plt.title("Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.grid(True)
    return fig