#pragma once

#include <omp.h>
#include <cmath>
//TODO::Do time_test with different number of cpus



const double DT = 0.5e-4;
const int NUM_PARTICLES = 500; //4 * num_cells_per_dim ^ 3
const double TEMPERATURE = 2.0;
const double CELL_SIZE = pow(NUM_PARTICLES, 1.0 / 3);
const int NUM_FRAMES = 100;
const int ITERS_PER_FRAME = 2000;
const double CUT_DIST = 20;
const double TAU = 5.0 / 8;