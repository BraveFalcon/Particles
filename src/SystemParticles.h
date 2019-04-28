#pragma once

#include "Vector3.hpp"
#include "../model_constants.h"
#include "Vector3.hpp"
#include <random>
#include <stdio.h>

class SystemParticles {
private:
    Vector3d *poses;
    Vector3d *prev_poses;
    Vector3d *vels;
    Vector3d *forces;
    bool is_energy_actual;

    Vector3d calc_near_r(const Vector3d &pos1, const Vector3d &pos2);

    Vector3d calc_force(const Vector3d &pos1, const Vector3d &pos2);

    void update_forces();

    std::pair<double, double> mean_std_kin(double tau);

public:
    explicit SystemParticles(unsigned seed);
    //TODO::Gen vels on Maxwell distribution

    ~SystemParticles();

    void update_state(int num_iters);

    void write_bin(FILE *file);

    double get_energy();

    double get_kin_energy();
};

