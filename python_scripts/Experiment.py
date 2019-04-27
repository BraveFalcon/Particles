import numpy as np
from os import path
import sys
import struct
import ctypes

sys.path.append(path.split(path.split(path.abspath(__file__))[0])[0])
from python_scripts import CLib


class Experiment:
    temperature = None

    def __init__(self, file_path):
        if file_path.split('.')[-1] == "bin":
            file = open(file_path, "rb")
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
            self.ts = np.linspace(0, self.num_frames * time_per_frame, self.num_frames, endpoint=False)

            bin_parser = CLib._clib.parse_data_file
            bin_parser.argtypes = [
                ctypes.POINTER(ctypes.c_char),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double)
            ]
            bin_parser(
                file_path.encode('ascii'),
                self.poses.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                self.vels.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                self.energies.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            )

            self.temperature = 1 / 3 * np.mean(np.sum(np.linalg.norm(self.vels, axis=2) ** 2, axis=1)) \
                               / self.num_particles

    def cut(self, _slice):
        pass

    def save(self, savePath):
        pass