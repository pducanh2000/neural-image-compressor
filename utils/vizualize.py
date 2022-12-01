import os
import numpy as np
import matplotlib.pyplot as plt


def plot_curve(config_params, name, nll_val, metric_name='loss'):
    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel(metric_name)
    save_result_path = os.path.join(config_params["result"], name + metric_name + '_val_curve.pdf')
    plt.savefig(save_result_path, bbox_inches='tight')
    plt.close()
