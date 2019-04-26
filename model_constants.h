#pragma once

#include <string>

//TODO::Go to concentration and temperature params, delete force_cut_dist, replace save_path to arguments

const double DT = 5e-4;
const int NUM_PARTICLES = 1000;
const double MAX_INIT_VEL = 3.5;
const double CELL_SIZE = 50;
const double TIME_MODELING = 3000;
const double TIME_PER_FRAME = 0.05;
const std::string SAVE_PATH = "/home/brave_falcon/Particles_Experiments/new";
