import numpy as np
import math
from os import path
import sys

sys.path.append(path.split(path.split(path.abspath(__file__))[0])[0])
from python_scripts.experiment import Experiment
from python_scripts import results


def gen_results(experiment, save_path):
    results.put("E_full", np.mean(experiment.energies), np.std(experiment.energies), save_path)

    kin_energies = np.sum(np.linalg.norm(experiment.vels, axis=2) ** 2 / 2, axis=1)
    mean_kin_energy = np.mean(kin_energies)
    std_kin_energy = np.std(kin_energies)
    results.put("E_kin/E_full", mean_kin_energy / np.mean(experiment.energies), 0, save_path)

    results.put("temperature",
                2 / 3 * mean_kin_energy / experiment.num_particles,
                2 / 3 * std_kin_energy / experiment.num_particles, save_path)

    results.put("pressure", np.mean(experiment.pressures), np.std(experiment.pressures), save_path)


if __name__ == "__main__":
    experiment_path = sys.argv[1]

    gen_results(Experiment(experiment_path), experiment_path)
