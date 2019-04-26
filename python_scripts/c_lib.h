#pragma once

#include <stdlib.h>
#include <math.h>
#include <stdio.h>

void diffusion(
        int num_frames,
        int num_particles,
        const double *poses,
        double *res
);

void radial_distr(
        int num_frames,
        int num_particles,
        const double *poses,
        double cell_size,
        double *res,
        int num_bins,
        double max_r
);

void parse_data_file(
        const char *file_path,
        double *poses,
        double *vels,
        double *energies
);