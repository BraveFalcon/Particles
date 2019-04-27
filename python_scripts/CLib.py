import ctypes
from os import path
import numpy as np

_clib = ctypes.CDLL(path.join(path.abspath(path.dirname(__file__)), 'c_lib.so'))


def diffusion(experiment):
    cfunc = _clib.diffusion
    cfunc.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double)
    ]

    sqrs_delta_r = np.empty(experiment.num_frames)
    cfunc(
        ctypes.c_int(experiment.num_frames),
        ctypes.c_int(experiment.num_particles),
        experiment.poses.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sqrs_delta_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )

    return sqrs_delta_r


def radial_distr(experiment, cut_dist=None, num_bins=None):
    if cut_dist is None:
        cut_dist = experiment.cell_size / 2
    if num_bins is None:
        num_bins = int(
            experiment.num_particles * (cut_dist / experiment.cell_size) ** 3 * np.sqrt(experiment.num_frames)
        )

    cfunc = _clib.radial_distr
    cfunc.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double
    ]

    hs = np.empty(num_bins, dtype='float64')
    bins = np.linspace(0, cut_dist, num_bins, endpoint=False)
    cfunc(
        experiment.num_frames,
        experiment.num_particles,
        experiment.poses.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        experiment.cell_size,
        hs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        num_bins,
        cut_dist
    )

    return bins, hs
