import numpy as np
from os import path
import sys
import struct
import ctypes
import pickle

sys.path.append(path.split(path.split(path.abspath(__file__))[0])[0])
from python_scripts import CLib


class Experiment:
    __dump_name = "experiment.pickle"

    def save_pickle(self, save_path):
        file = open(path.join(save_path, self.__dump_name), "wb")
        pickle.dump(self, file, protocol=4)
        file.close()

    def __load_pickle(self, experiment_path):
        file = open(path.join(experiment_path, self.__dump_name), "rb")
        self.__dict__ = pickle.load(file).__dict__
        file.close()

    def __get_temperature(self):
        return 1 / 3 * np.mean(np.mean(np.linalg.norm(self.vels, axis=2) ** 2, axis=1))

    def __init__(self, experiment_path):
        if path.exists(path.join(experiment_path, self.__dump_name)):
            self.__load_pickle(experiment_path)
        elif path.exists(path.join(experiment_path, 'data.bin')):
            file = open(path.join(experiment_path, 'data.bin'), "rb")
            self.num_frames, self.num_particles, time_per_frame, self.cell_size = struct.unpack(
                "=iidd",
                file.read(struct.calcsize("=iidd"))
            )
            file.close()

            if not 0 < self.num_frames < 1e5 or not 0 < self.num_particles < 1e5 or time_per_frame < 0 or self.cell_size < 0:
                print("Bad binary file")
                print("Num_particles: %i" % self.num_particles)
                print("Num_frames: %i" % self.num_particles)
                print("Time per frame: %d" % time_per_frame)
                print("Cell size: %d" % self.cell_size)
                exit(1)

            self.poses = np.empty((self.num_frames, self.num_particles, 3), dtype='float64')
            self.vels = np.empty((self.num_frames, self.num_particles, 3), dtype='float64')
            self.energies = np.empty(self.num_frames, dtype='float64')
            self.pressures = np.empty(self.num_frames, dtype='float64')
            self.ts = np.linspace(time_per_frame, self.num_frames * time_per_frame, self.num_frames)

            bin_parser = CLib._clib.parse_data_file
            bin_parser.argtypes = [
                ctypes.POINTER(ctypes.c_char),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double)
            ]
            bin_parser(
                path.join(experiment_path, "data.bin").encode('ascii'),
                self.poses.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                self.vels.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                self.energies.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                self.pressures.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            )
        else:
            dump = open(path.join(experiment_path, "dump.txt"), "r")
            [dump.readline() for _ in range(3)]
            self.num_particles = int(dump.readline())
            dump.readline()
            x_l, x_r = map(float, dump.readline().split())
            self.cell_size = x_r - x_l
            dump.seek(0)
            output = open(path.join(experiment_path, "output.txt"), "r")
            self.num_frames, time_per_frame = map(float, output.readline().split(':'))
            self.num_frames = int(self.num_frames)
            self.energies = np.array([float(line) for line in output], dtype='float64')
            self.pressures = np.zeros(self.num_frames, dtype='float64')
            self.ts = np.linspace(time_per_frame, self.num_frames * time_per_frame, self.num_frames)
            self.poses = np.empty((self.num_frames, self.num_particles, 3), dtype='float64')
            self.vels = np.empty((self.num_frames, self.num_particles, 3), dtype='float64')
            [dump.readline() for _ in range(9 + self.num_particles)]
            for frame in range(self.num_frames):
                [dump.readline() for _ in range(9)]
                for _ in range(self.num_particles):
                    words = dump.readline().split()
                    particle = int(words[0]) - 1
                    self.poses[frame, particle] = np.array(words[1:4], dtype='float64')
                    self.vels[frame, particle] = np.array(words[4:], dtype='float64')
            output.close()
            dump.close()
        self.temperature = self.__get_temperature()
        self.unbound()
        self.save_pickle(experiment_path)

    def unbound(self):
        for particle in range(self.num_particles):
            for dim in range(3):
                for frame in range(1, self.num_frames):
                    delta = self.poses[frame, particle, dim] - self.poses[frame - 1, particle, dim] - \
                            self.vels[frame - 1, particle, dim] * self.ts[0]
                    if abs(delta) > 0.8 * self.cell_size:
                        if delta > 0:
                            self.poses[frame:, particle, dim] -= self.cell_size
                        else:
                            self.poses[frame:, particle, dim] += self.cell_size

    def cut(self, _slice):
        self.poses = self.poses[_slice]
        self.vels = self.vels[_slice]
        self.energies = self.energies[_slice]
        self.num_frames = len(self.energies)
        self.ts = self.ts[_slice]
        self.ts -= self.ts[0]
        self.temperature = self.__get_temperature()

    def save_xyz(self, save_path):
        file = open(path.join(save_path, "data.xyz"), "w")
        for frame in range(self.num_frames):
            print(self.num_particles, file=file)
            print(f"Lattice=\"{self.cell_size} 0 0 0 {self.cell_size} 0 0 0 {self.cell_size}\" Properties=pos:R:3",
                  file=file)
            for particle in range(self.num_particles):
                print("%.3f\t%.3f\t%.3f\t%.3f" % (
                        tuple((self.poses[frame, particle] + self.cell_size / 2) % self.cell_size) + (
                    np.linalg.norm(self.vels[frame, particle]) ** 2 / 3,)), file=file)
        file.close()
