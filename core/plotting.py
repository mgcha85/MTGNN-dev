# core/plotting.py
import numpy as np
import matplotlib.pyplot as plt

def plot_predicted_actual(predicted, actual, title, out_png, out_pdf, variance=None, conf95=None):
    """
    predicted, actual: 1D np.array (T,)
    """
    x = np.arange(1, len(predicted) + 1)
    plt.plot(x, actual, 'b-', label='Actual')
    plt.plot(x, predicted, '--', color='purple', label='Predicted')
    if conf95 is not None:
        plt.fill_between(x, predicted - conf95, predicted + conf95, alpha=0.5, color='pink', label='95% Confidence')
    plt.legend(loc="best")
    plt.grid(True)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight", format='pdf')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
