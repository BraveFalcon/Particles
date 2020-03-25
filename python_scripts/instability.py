import sys
from os import path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

sys.path.append(path.split(path.split(path.abspath(__file__))[0])[0])
from python_scripts.experiment import Experiment


def get_figure(exp1, exp2):
    rs = np.mean(np.linalg.norm(exp1.poses - exp2.poses, axis=2) ** 2, axis=1) / exp1.cell_size
    vs = np.mean(np.linalg.norm(exp1.vels - exp2.vels, axis=2) ** 2, axis=1) / np.sqrt(3 * np.mean(exp1.temperature))

    # popt, pcov = curve_fit(lambda x, a, b: a * np.exp(b * x),
    #                       xdata=exp1.ts, ydata=)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.subplots_adjust(bottom=0.1, left=0.05, right=0.95, top=0.92)
    fig.suptitle("...", fontsize=20)

    ax.set_xlabel(r"$t$, ", fontsize=14)
    ax.ticklabel_format(style='sci', scilimits=(-2, 2))
    ax.set_yscale('log')

    ax.plot(exp1.ts, rs, 's', label=r'$\left\langle \Delta r^2 \right\rangle$')
    ax.plot(exp1.ts, vs, '^', label=r'$\left\langle \Delta v^2 \right\rangle$')

    ax.legend()

    return fig


if __name__ == "__main__":
    experiment1_path = sys.argv[1]
    experiment2_path = sys.argv[2]

    experiment1 = Experiment(experiment1_path)
    experiment2 = Experiment(experiment2_path)
    fig = get_figure(experiment1, experiment2)
    plt.show()
