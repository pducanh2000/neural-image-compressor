import os
import numpy as np
import matplotlib.pyplot as plt


def plot_curve(config_params, metric_name, metric_log):
    plt.plot(np.arange(len(metric_log)), metric_log, linewidth="3")
    plt.xlabel("epochs")
    plt.ylabel(metric_name)
    result_model_folder = os.path.join(config_params["result_folder"], config_params["model_name"])
    if not os.path.isdir(result_model_folder):
        os.mkdir(result_model_folder)
    save_result_path = os.path.join(result_model_folder, metric_name + "_curve.pdf")
    plt.savefig(save_result_path, bbox_inches="tight")
    plt.close()
