import numpy as np
import math
from os import path
import sys

sys.path.append(path.split(path.split(path.abspath(__file__))[0])[0])
from python_scripts.experiment import Experiment


def __str_result(name, value, error):
    exp = int(("%e" % value).split('e')[1])
    value /= 10 ** exp
    error /= 10 ** exp
    error_exp = abs(int(("%e" % error).split('e')[1]) - 1)
    return "{0} = (%.{2}f +- %.{2}f)e{1}".format(name, exp, error_exp) % (value, error)


def gen_results(experiment, save_path):
    file = open(path.join(save_path, 'results.txt'), "w")
    results = list()

    results.append(("full_energy", np.mean(experiment.energies), np.std(experiment.energies)))

    kin_energies = np.sum(np.linalg.norm(experiment.vels, axis=2) ** 2 / 2, axis=1)
    mean_kin_energy = np.mean(kin_energies)
    std_kin_energy = np.std(kin_energies)
    results.append(("kin_energy", mean_kin_energy, std_kin_energy))

    results.append(("temperature",
                    2 / 3 * mean_kin_energy / experiment.num_particles,
                    2 / 3 * std_kin_energy / experiment.num_particles)
                   )
    for name, value, error in results:
        print(__str_result(name, value, error), file=file)
    file.close()


if __name__ == "__main__":
    experiment_path = sys.argv[1]

    gen_results(Experiment(experiment_path), experiment_path)
