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
    Vector3d *forces;
    double model_time;
    bool is_energy_actual;

    Vector3d calc_near_r(const Vector3d &pos1, const Vector3d &pos2);

    Vector3d calc_force(const Vector3d &pos1, const Vector3d &pos2);

    void update_forces();

public:
    explicit SystemParticles(unsigned seed);

    ~SystemParticles();

    void update_state(double time);

    void write_bin(FILE *out);

    double get_model_time();

    double get_energy();
};

