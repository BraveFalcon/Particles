#include <iostream>
#include <fstream>
#include "Vector3.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <omp.h>
#include <chrono>

const double DT = 1e-3;
const double CONCENTRATION = 0.125;
const unsigned NUMBER_PARTICLES = 1000;
const double MAX_INIT_VEL = 5;

const double CELL_SIZE = std::pow(NUMBER_PARTICLES / CONCENTRATION, 1.0 / 3); //сторона куба
const double MASS = 1;

struct Particle {
    Vector3d pos;
    Vector3d vel;
};

void print_data(FILE *out, const Particle *particles, const std::string &comment) {
    fprintf(out, "%u\n", NUMBER_PARTICLES);
    fprintf(out, "%s\n", comment.c_str());
    for (int i = 0; i < NUMBER_PARTICLES; ++i)
        fprintf(out, "%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n", particles[i].pos.x, particles[i].pos.y, particles[i].pos.z,
                particles[i].vel.x, particles[i].vel.y, particles[i].vel.z);
}

void set_rand_state(Particle *particles) {
    double max_poj_vel = MAX_INIT_VEL / std::sqrt(3);
    std::uniform_real_distribution<double> unif(-max_poj_vel, max_poj_vel);
    std::default_random_engine re;
    re.seed(42);
    for (int i = 0; i < NUMBER_PARTICLES; ++i) {
        particles[i].vel.set_values(unif(re), unif(re), unif(re));
    }


    const int a = std::ceil(pow(NUMBER_PARTICLES, 1 / 3.0)); //количество частиц на одно измерение куба
    const double dist = CELL_SIZE / (a + 1) + 1e-6;

    int i = 0;
    for (double x = -CELL_SIZE / 2 + dist; x < CELL_SIZE / 2; x += dist)
        for (double y = -CELL_SIZE / 2 + dist; y < CELL_SIZE / 2; y += dist)
            for (double z = -CELL_SIZE / 2 + dist; z < CELL_SIZE / 2; z += dist) {
                if (i >= NUMBER_PARTICLES)
                    return;
                particles[i].pos.set_values(x, y, z);
                ++i;
            }
}

Vector3d calc_near_r(const Vector3d &pos1, const Vector3d &pos2) {
    Vector3d res = pos2 - pos1;
    for (int i = 0; i < 3; ++i)
        if (res[i] > 0.5 * CELL_SIZE)
            res[i] -= CELL_SIZE;
        else if (res[i] < -0.5 * CELL_SIZE)
            res[i] += CELL_SIZE;

    return res;
}

void calc_forces(const Particle *particles, Vector3d *forces) {
    for (int i = 0; i < NUMBER_PARTICLES; ++i)
        forces[i].set_values(0.0);

    for (int i = 0; i < NUMBER_PARTICLES; ++i)
        for (int j = i + 1; j < NUMBER_PARTICLES; ++j) {
            Vector3d r_near = calc_near_r(particles[i].pos, particles[j].pos);
            double dist_sqr = r_near.sqr();
            Vector3d force = r_near * 24 * (2 * pow(dist_sqr, -7) - pow(dist_sqr, -4));
            forces[i] -= force;
            forces[j] += force;
        }
}

Vector3d calc_force(const Vector3d &pos1, const Vector3d &pos2) {
    Vector3d r_near = calc_near_r(pos1, pos2);
    double dist_sqr = r_near.sqr();
    return r_near * 24 * (2 * pow(dist_sqr, -7) - pow(dist_sqr, -4));
}


void calc_forces_PAR(const Particle *particles, Vector3d *forces) {
#pragma omp parallel for
    for (int i = 0; i < NUMBER_PARTICLES; ++i) {
        forces[i].set_values(0.0);

        for (int j = 0; j < i; ++j)
            forces[i] -= calc_force(particles[i].pos, particles[j].pos);

        for (int j = i + 1; j < NUMBER_PARTICLES; ++j)
            forces[i] -= calc_force(particles[i].pos, particles[j].pos);

    }
}


void update_state(Particle *particles) {
    static Vector3d forces[NUMBER_PARTICLES];
    calc_forces_PAR(particles, forces);
    for (int i = 0; i < NUMBER_PARTICLES; ++i) {
        particles[i].vel += forces[i] / MASS * DT;
        particles[i].pos += particles[i].vel * DT;
        particles[i].pos = calc_near_r(Vector3d(0), particles[i].pos);
    }
}

double calc_energy(const Particle *particles) {
    double energy = 0;
    for (int i = 0; i < NUMBER_PARTICLES; ++i) {
        energy += 0.5 * MASS * particles[i].vel.sqr();
        for (int j = i + 1; j < NUMBER_PARTICLES; ++j) {
            double dist_sqr = calc_near_r(particles[i].pos, particles[j].pos).sqr();
            energy += 4 * (pow(dist_sqr, -6) - pow(dist_sqr, -3));
        }
    }
    return energy;
}

double calc_energy_PAR(const Particle *particles) {
    double energy = 0;
#pragma omp parallel for reduction(+:energy)
    for (int i = 0; i < NUMBER_PARTICLES; ++i) {
        energy += 0.5 * MASS * particles[i].vel.sqr();
        for (int j = i + 1; j < NUMBER_PARTICLES; ++j) {
            double dist_sqr = calc_near_r(particles[i].pos, particles[j].pos).sqr();
            energy += 4 * (pow(dist_sqr, -6) - pow(dist_sqr, -3));
        }
    }
    return energy;
}

void calc_and_write_data(Particle *particles, const double time_modeling, const double time_per_frame) {
    FILE *info_file = fopen("./info.txt", "w");
    fprintf(info_file,
            "DT = %e\nCONCENTRATION = %f\nNUMBER_PARTICLES = %d\nMAX_INIT_VEL = %f\nMASS = %f\nCELL_SIZE = %f\nTIME_MODELING = %f\nTIME_PER_FRAME = %f\n",
            DT, CONCENTRATION, NUMBER_PARTICLES, MAX_INIT_VEL, MASS, CELL_SIZE, time_modeling, time_per_frame);
    fclose(info_file);
    FILE *out_data_file = fopen("./data.xyz", "w");

    double global_time = 0;

    unsigned n = 0;
    double sum_energies = 0;

    std::chrono::high_resolution_clock::time_point start, end, cur;
    start = std::chrono::high_resolution_clock::now();
    while (global_time < time_modeling) {

        double frac_done = global_time / time_modeling;
        cur = std::chrono::high_resolution_clock::now();
        double time_left = std::chrono::duration_cast<std::chrono::duration<double>>(cur - start).count() / frac_done *
                           (1 - frac_done);

        double cur_energy = calc_energy_PAR(particles);
        sum_energies += cur_energy;
        n++;
        double mean_energy = sum_energies / n;

        printf("\r%.2f %% (%.0f s left). Energy deviation: %.0e", frac_done * 100, time_left,
               std::abs(1 - cur_energy / mean_energy));
        std::cout.flush();

        print_data(out_data_file, particles, std::to_string(global_time) + "\t" + std::to_string(cur_energy));
        double frame_time = 0;
        while (frame_time < time_per_frame && global_time < time_modeling) {
            update_state(particles);
            global_time += DT;
            frame_time += DT;
        }
    }
    double cur_energy = calc_energy_PAR(particles);
    print_data(out_data_file, particles, std::to_string(global_time) + "\t" + std::to_string(cur_energy));

    end = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "\rCalculation time: " << time_span.count() << "s\n";

    fclose(out_data_file);
    system("python3 ../python_scripts/make_dirs_and_images.py");
}

int main() {
    omp_set_num_threads(omp_get_num_procs());

    const double time_modeling = 10;

    Particle particles[NUMBER_PARTICLES];
    set_rand_state(particles);

    calc_and_write_data(particles, time_modeling, 0.01);

}