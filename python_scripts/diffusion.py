import sys
from os import path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

sys.path.append(path.split(path.split(path.abspath(__file__))[0])[0])
from python_scripts.experiment import Experiment
from python_scripts import CLib, results


def get_figure(exp):
    sqrs_delta_r = CLib.diffusion(exp)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.suptitle('diffusion', fontsize=20)
    fig.subplots_adjust(bottom=0.1, left=0.1, right=0.95, top=0.90)
    ax.set_xlabel('time, $\sigma \sqrt{m / \epsilon}$', fontsize=14)
    ax.set_ylabel(r'$\langle \Delta r^2 \rangle, \sigma^2$', fontsize=14)
    ax.ticklabel_format(style='sci', scilimits=(-1, 3))
    ax.set_xlim((exp.ts[0], exp.ts[-1]))
    ax.set_ylim((0, np.max(sqrs_delta_r)))

    ax.plot(exp.ts, sqrs_delta_r, 'ro', ms=2)

    start = int(len(exp.ts) * 0.2)
    xs = exp.ts[start:]
    ys = sqrs_delta_r[start:]
    popt, pcov = curve_fit(lambda x, a, b: a * x + b,
                           xdata=xs, ydata=ys)
    results.put("Diffusion", popt[0], (pcov[0, 0]) ** 0.5)
    ax.plot(exp.ts, popt[0] * exp.ts + popt[1], 'g--', lw=2)

    return fig


if __name__ == '__main__':
    experiment_path = sys.argv[1]

    experiment = Experiment(experiment_path)
    fig = get_figure(experiment)
    plt.show()
