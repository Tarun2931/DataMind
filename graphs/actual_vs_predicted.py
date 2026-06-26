import matplotlib.pyplot as plt
def create_actual_vs_predicted(actual,predicted):
    fig = plt.figure(figsize=(7, 7))
    plt.scatter(actual,predicted,alpha=0.7)
    plt.plot([min(actual), max(actual)],[min(actual), max(actual)],color="red",linewidth=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    plt.grid(True)
    return fig