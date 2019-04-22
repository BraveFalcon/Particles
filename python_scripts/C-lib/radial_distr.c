#include <malloc.h>
#include <math.h>
#include <stdio.h>

void
get_distr(const double *data,
          int num_frames,
          int num_particles,
          double cell_size,
          double *res,
          int num_bins,
          double max_r) {
    int sizeof_particle = 6 * sizeof(double);
    int sizeof_frame = num_particles * sizeof_particle;
    unsigned long long *ns = malloc(num_bins * sizeof(unsigned long long));
    for (int i = 0; i < num_bins; ++i)
        ns[i] = 0;
    for (int frame = 0; frame < num_frames; ++frame) {
        for (int particle = 0; particle < num_particles; ++particle) {
            const double *pos = (void *) (data) + frame * sizeof_frame + particle * sizeof_particle;
            for (int pair_particle = particle + 1; pair_particle < num_particles; ++pair_particle) {
                double r_sqr = 0;
                const double *pair_pos = (void *) data + frame * sizeof_frame + pair_particle * sizeof_particle;
                for (int axis = 0; axis < 3; ++axis) {
                    double proj = pos[axis] - pair_pos[axis];
                    proj -= round(proj / cell_size) * cell_size;
                    r_sqr += proj * proj;
                }
                double r = sqrt(r_sqr);
                if (r < max_r) {
                    unsigned bin = (unsigned) (r / max_r * num_bins);
                    ++ns[bin];
                }
            }
        }
    }
    double concentration = num_particles / (pow(cell_size, 3));
    for (int bin = 0; bin < num_bins; ++bin) {
        res[bin] = 2 * ns[bin] / (4 * M_PI * pow(max_r / num_bins, 3) * (bin * bin + bin + 1.0 / 3)) / num_particles /
                   num_frames / concentration;
    }
}
