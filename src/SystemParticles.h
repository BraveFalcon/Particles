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

    Vector3d calc_near_r(const Vector3d &pos1, const Vector3d &pos2) const;

    Vector3d calc_force(const Vector3d &pos1, const Vector3d &pos2) const;

    void update_forces();

public:
    explicit SystemParticles(unsigned seed);

    ~SystemParticles();

    void update_state(int num_iters);

    void write_bin(FILE *file) const;

    double get_energy() const;

    double get_temperature() const;
};

