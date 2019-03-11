import numpy as np
from os import path

def resave_xyz_to_npy(infile, outdir):
    '''
    :param filename: full name of xyz file, which contains 6 columns: x y z vx vy vz
    :return: ndarray with shape (num_frames, num_particles, 2, 3),
    where data[i_frame, i_particle] contains array(pos, vel), where pos and vel - 3d vectors
    '''

    file = open(infile, 'r')
    data = []
    ts = []
    energies = []
    first_line = file.readline()
    n = int(first_line)
    while first_line != '':
        time, energy = map(float, file.readline().split())
        ts.append(time)
        energies.append(energy)
        frame_data = []
        for _ in range(n):
            particle = file.readline()
            x, y, z, vx, vy, vz = map(float, particle.split())
            pos = np.array((x, y, z))
            vel = np.array((vx, vy, vz))
            frame_data.append(np.array((pos, vel)))
        data.append(np.array(frame_data))
        first_line = file.readline()
    np.save(path.join(outdir, "data"), data)
    np.save(path.join(outdir, "ts"), ts)
    np.save(path.join(outdir, "energies"), energies)

