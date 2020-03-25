#include "c_lib.h"

void diffusion(
        int num_frames,
        int num_particles,
        const double *poses,
        double *res) {
    res[0] = 0.0;
    for (int tau = 1; tau < num_frames; ++tau) {
        res[tau] = 0.0;
        for (int start_point = 0; start_point < num_frames - tau; start_point += tau)
            for (int particle = 0; particle < num_particles; ++particle) {
                const double *pos1 = poses + 3 * (num_particles * start_point + particle);
                const double *pos2 = poses + 3 * (num_particles * (start_point + tau) + particle);
                for (int axis = 0; axis < 3; ++axis) {
                    double proj = pos2[axis] - pos1[axis];
                    res[tau] += proj * proj;
                }
            }
        res[tau] /= num_particles * ((num_frames - 1) / tau);
    }
}


void radial_distr(
        int num_frames,
        int num_particles,
        const double *poses,
        double cell_size,
        double *res,
        int num_bins,
        double max_r) {
    omp_set_num_threads(omp_get_num_procs());
    unsigned int ns[omp_get_max_threads()][num_bins];
    if (ns == NULL) {
        fprintf(stderr, "Malloc failed in radial_distr c-func\n");
        exit(1);
    }
    for (int i = 0; i < num_bins; ++i)
        for (int thid = 0; thid < omp_get_max_threads(); ++thid)
            ns[thid][i] = 0;
#pragma omp parallel for
    for (int frame = 0; frame < num_frames; ++frame) {
        for (int particle = 0; particle < num_particles; ++particle) {
            const double *pos = poses + 3 * (num_particles * frame + particle);
            for (int pair_particle = particle + 1; pair_particle < num_particles; ++pair_particle) {
                double r_sqr = 0;
                const double *pair_pos = poses + 3 * (num_particles * frame + pair_particle);
                for (int axis = 0; axis < 3; ++axis) {
                    double proj = pos[axis] - pair_pos[axis];
                    proj -= round(proj / cell_size) * cell_size;
                    r_sqr += proj * proj;
                }
                double r = sqrt(r_sqr);
                if (r < max_r) {
                    unsigned bin = (unsigned) (r / max_r * num_bins);
                    ++ns[omp_get_thread_num()][bin];
                }
            }
        }
    }
    double concentration = num_particles / (pow(cell_size, 3));
    for (int i = 0; i < num_bins; ++i)
        for (int thid = 1; thid < omp_get_max_threads(); ++thid)
            ns[0][i] += ns[thid][i];
    for (int bin = 0; bin < num_bins; ++bin) {
        res[bin] =
                2 * ns[0][bin] / (4 * M_PI * pow(max_r / num_bins, 3) * (bin * bin + bin + 1.0 / 3)) / num_particles /
                num_frames / concentration;
    }
}

void parse_data_file(
        const char *file_path,
        double *poses,
        double *vels,
        double *energies,
        double *pressures) {
    FILE *file = fopen(file_path, "rb");
    if (!file) {
        fprintf(stderr, "Can't open file in parse_data_file c-func");
        exit(1);
    }
    int num_frames;
    if (!fread(&num_frames, sizeof(int), 1, file)) {
        fprintf(stderr, "Can't read num_frames from bin_file in parse_data_file c-func");
        exit(1);
    }
    int num_particles;
    if (!fread(&num_particles, sizeof(int), 1, file)) {
        fprintf(stderr, "Can't read num_particles from bin_file in parse_data_file c-func");
        exit(1);
    }
    if (fseek(file, 2 * sizeof(double), SEEK_CUR) != 0) {
        fprintf(stderr, "Can't read time_per_frame or cell_size from bin_file in parse_data_file c-func");
        exit(1);
    }
    for (int frame = 0; frame < num_frames; ++frame) {
        if (!fread(energies + frame, sizeof(double), 1, file)) {
            fprintf(stderr, "Can't read energy of %d frame in parse_data_file c-func", frame);
            exit(1);
        }
        if (!fread(pressures + frame, sizeof(double), 1, file)) {
            fprintf(stderr, "Can't read pressure of %d frame in parse_data_file c-func", frame);
            exit(1);
        }
        if (fread(poses + frame * 3 * num_particles, sizeof(double), 3 * num_particles, file) != 3 * num_particles) {
            fprintf(stderr, "Can't read poses of %d frame in parse_data_file c-func", frame);
            exit(1);
        }
        if (fread(vels + frame * 3 * num_particles, sizeof(double), 3 * num_particles, file) != 3 * num_particles) {
            fprintf(stderr, "Can't read vels of %d frame in parse_data_file c-func", frame);
            exit(1);
        }
    }
}