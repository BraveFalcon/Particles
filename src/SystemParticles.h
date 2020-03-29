#pragma once

#include "Vector3.hpp"
#include "TimeLeft.hpp"
#include <vector>
#include <algorithm>
#include <utility>
#include <random>
#include <cstdio>
#include <omp.h>
#include "Value.hpp"
#include "LogStream.h"
#include <iomanip>


class SystemParticles {
private:
    Vector3d *poses;
    Vector3d *vels;
    Vector3d *forces;
    Vector3d *prev_forces;
    Vector3d **th_forces;
    double **th_data;
    double virial;
    double pot_energy;
    const double CUT_DIST = 2.5;
    const int NUM_THREADS = 4;
    LogStream &log;


    [[nodiscard]] Vector3d get_near_r(const Vector3d &pos1, const Vector3d &pos2) const;
    
    [[nodiscard]] double calc_rel_std_energy(double DT, double time);

    void update_forces();

public:
    double dt = 1e-3;
    const int NUM_PARTICLES;
    double cell_size;

    void init_arrays();

    explicit SystemParticles(int num_cells_per_dim, double density, LogStream &logStream);

    explicit SystemParticles(std::string file_path, int frame, int num_particles, LogStream &logStream);

    ~SystemParticles();

    void update_state(int num_iters);

    void termostat_berendsen(int num_iters, double temp, double tau);

    void termostat_berendsen(double temp);

    void npt_berendsen(double press, double temp, double mean_beta = 1.0);

    void npt_berendsen(unsigned num_iters, double press, double temp, double tau, double beta);

    void termostat_andersen(int num_particles, double temp, int seed = 42);

    void init_bin_file(FILE *file, int num_frames, double time_per_frame) const;

    void write_bin_file(FILE *file) const;

    [[nodiscard]] double get_energy() const;

    [[nodiscard]] double get_temperature() const;

    [[nodiscard]] double get_pressure() const;

    [[nodiscard]] double get_free_time() const;

    void guess_dt(double iter_time);

    void set_vels(double temperature, unsigned seed = 42);

    std::string print_info(double frac_done) const;

};

