import matplotlib.pyplot as plt
def create_roc_curve(fpr,tpr,roc_auc):
    fig = plt.figure(figsize=(7,6))
    plt.plot(fpr,tpr,linewidth=2,label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    return fig