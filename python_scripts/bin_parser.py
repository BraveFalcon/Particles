import numpy as np
import struct
import sys
from os import path


def read_file(file_path):
    '''
    :param file_path: path to binary file, which contains 6 columns: x y z vx vy vz
    :return: data, energies, ts,
    where data is ndarray with shape (num_frames, num_particles, 2, 3),
    where data[i_frame, i_particle] contains array(pos, vel), where pos and vel - 3d vectors
    '''
    file = open(file_path, 'rb')
    num_particles, time_per_frame = struct.unpack('=id', file.read(4 + 8))
    read_size = 8 + 48 * num_particles
    data = []
    energies = []

    buffer = file.read(read_size)
    while buffer:
        if len(buffer) != read_size:
            print("Bad file, cant read it")
            exit(1)
        offset = 0
        energies.append(struct.unpack_from('d', buffer, offset)[0])
        offset += 8
        frame_data = []
        for _ in range(num_particles):
            x, y, z, vx, vy, vz = struct.unpack_from('6d', buffer, offset)
            offset += 6 * 8
            pos = np.array((x, y, z))
            vel = np.array((vx, vy, vz))
            frame_data.append(np.array((pos, vel)))
        data.append(np.array(frame_data))
        buffer = file.read(read_size)
    num_frames = len(energies)
    ts = np.linspace(0, (num_frames - 1) * time_per_frame, num_frames)
    return np.array(data), np.array(energies), ts


def save_data_to_xyz(data, cell_size, outfile_path):
    outfile = open(outfile_path, 'w')
    for frame in range(data.shape[0]):
        print(data.shape[1], file=outfile)
        print(f"Lattice=\"{cell_size} 0 0 0 {cell_size} 0 0 0 {cell_size}\" Properties=pos:R:3", file=outfile)
        for particle in data[frame]:
            print("%.3e\t%.3e\t%.3e" % tuple((particle[0] + cell_size / 2) % cell_size),
                  file=outfile)


def parse_info_file(info_file_path):
    file = open(info_file_path, 'r')
    params = dict()
    for line in file:
        name, value = line.split()
        params[name] = float(value)
    return params


if __name__ == "__main__":
    experiment_path = sys.argv[1]

    data, energies, ts = read_file(path.join(experiment_path, 'data.bin'))
    cell_size = parse_info_file(path.join(experiment_path, 'info.txt'))['CELL_SIZE']
    save_data_to_xyz(data, cell_size, sys.argv[2])
