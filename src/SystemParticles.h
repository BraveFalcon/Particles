#pragma once

#include "Vector3.hpp"
#include <random>
#include <cstdio>
#include <omp.h>


//TODO::Go to sdtl vectors
class SystemParticles {
private:
    Vector3d *poses;
    Vector3d *prev_poses;
    Vector3d *vels;
    Vector3d *forces;
    Vector3d **th_forces;
    double **th_data;
    const double CUT_DIST = 2.5;
    const int NUM_THREADS;

    [[nodiscard]] Vector3d calc_near_r(const Vector3d &pos1, const Vector3d &pos2) const;

    [[nodiscard]] double calc_virial(const Vector3d &pos1, const Vector3d &pos2) const;

    void update_forces();

public:
    double dt = 1e-3;
    const int NUM_PARTICLES;
    double CELL_SIZE;

    void init_arrays();

    explicit SystemParticles(int num_cells_per_dim, double density);

    explicit SystemParticles(std::string file_path, int frame, int num_particles);

    ~SystemParticles();

    void update_state(int num_iters);

    void termostat_berendsen(int num_iters, double temp, double tau);

    void termostat_andersen(int num_particles, double temp);

    void init_bin(FILE *file, int num_frames, double time_per_frame) const;

    void write_bin(FILE *file) const;

    [[nodiscard]] double get_energy() const;

    [[nodiscard]] double get_temperature() const;

    [[nodiscard]] double get_pressure() const;

    void set_vels(double temperature, unsigned seed = 42);
};

