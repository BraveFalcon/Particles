#include "SystemParticles.h"

Vector3d SystemParticles::calc_near_r(const Vector3d &pos1, const Vector3d &pos2) {
    Vector3d res = pos2 - pos1;
    res.x -= round(res.x / CELL_SIZE) * CELL_SIZE;
    res.y -= round(res.y / CELL_SIZE) * CELL_SIZE;
    res.z -= round(res.z / CELL_SIZE) * CELL_SIZE;
    return res;
}

Vector3d SystemParticles::calc_force(const Vector3d &pos1, const Vector3d &pos2) {
    Vector3d r_near = calc_near_r(pos1, pos2);
    double dist_sqr = r_near.sqr();
    return r_near * 24 * (2 * pow(dist_sqr, -7) - pow(dist_sqr, -4));
}

void SystemParticles::update_forces() {
#pragma omp parallel for
    for (int i = 0; i < NUMBER_PARTICLES; ++i) {
        forces[i].set_values(0.0);

        for (int j = 0; j < i; ++j)
            forces[i] -= calc_force(poses[i], poses[j]);

        for (int j = i + 1; j < NUMBER_PARTICLES; ++j)
            forces[i] -= calc_force(poses[i], poses[j]);
    }
}

double SystemParticles::get_energy() {
    static double energy;
    if (!is_energy_actual) {
        is_energy_actual = true;
        energy = 0;
#pragma omp parallel for reduction(+:energy)
        for (int i = 0; i < NUMBER_PARTICLES; ++i) {
            energy += 0.5 * ((poses[i] - prev_poses[i]) / DT).sqr();
            for (int j = i + 1; j < NUMBER_PARTICLES; ++j) {
                double dist_sqr = calc_near_r(poses[i], poses[j]).sqr();
                energy += 4 * (pow(dist_sqr, -6) - pow(dist_sqr, -3));
            }
        }
    }
    return energy;
}

void SystemParticles::update_state(double time) {
    is_energy_actual = false;
    double frame_time = 0;
    while (frame_time < time) {
        frame_time += DT;
        model_time += DT;
        update_forces();
        for (int i = 0; i < NUMBER_PARTICLES; ++i) {
            Vector3d new_pos = 2.0 * poses[i] - prev_poses[i] + forces[i] * DT * DT;
            prev_poses[i] = poses[i];
            poses[i] = new_pos;
        }
    }
}

void SystemParticles::write_bin(FILE *out) {
    double energy = get_energy();
    fwrite(&energy, sizeof(double), 1, out);
    for (int i = 0; i < NUMBER_PARTICLES; ++i) {
        Vector3d vel = (poses[i] - prev_poses[i]) / DT;
        fwrite(&poses[i], sizeof(Vector3d), 1, out);
        fwrite(&vel, sizeof(Vector3d), 1, out);
    }
}

double SystemParticles::get_model_time() { return model_time; }

SystemParticles::SystemParticles(unsigned seed) {
    poses = new Vector3d[NUMBER_PARTICLES];
    prev_poses = new Vector3d[NUMBER_PARTICLES];
    forces = new Vector3d[NUMBER_PARTICLES];
    model_time = 0.0;
    is_energy_actual = false;

    double max_poj_vel = MAX_INIT_VEL / std::sqrt(3);
    std::uniform_real_distribution<double> unif(-max_poj_vel, max_poj_vel);
    std::default_random_engine re;
    re.seed(seed);

    const double a = std::ceil(std::cbrt(NUMBER_PARTICLES)); //количество частиц на одно измерение куба
    const double dist = CELL_SIZE / a;

    int i = 0;
    for (double x = -CELL_SIZE / 2 + dist / 2; x < CELL_SIZE / 2 && i < NUMBER_PARTICLES; x += dist)
        for (double y = -CELL_SIZE / 2 + dist / 2; y < CELL_SIZE / 2 && i < NUMBER_PARTICLES; y += dist)
            for (double z = -CELL_SIZE / 2 + dist / 2; z < CELL_SIZE / 2 && i < NUMBER_PARTICLES; z += dist) {
                poses[i].set_values(x, y, z);
                ++i;
            }

    Vector3d mean_vel;
    for (i = 0; i < NUMBER_PARTICLES; ++i) {
        Vector3d vel = Vector3d(unif(re), unif(re), unif(re));
        prev_poses[i] = poses[i] - vel * DT;
        mean_vel += vel;
    }

    for (i = 0; i < NUMBER_PARTICLES; ++i)
        prev_poses[i] += mean_vel / NUMBER_PARTICLES * DT;
}

SystemParticles::~SystemParticles() {
    delete[] poses;
    delete[] prev_poses;
    delete[] forces;
}
