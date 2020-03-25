from os import path
import sys

sys.path.append(path.split(path.split(path.abspath(__file__))[0])[0])
from python_scripts.experiment import Experiment

if __name__ == "__main__":
    experiment_path = sys.argv[1]

    exp = Experiment(experiment_path)
    exp.save_xyz(experiment_path)
