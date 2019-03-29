#include <iostream>
#include "Vector3.hpp"
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include "../model_constants.h"
#include <omp.h>

struct Particle {
    Vector3d pos;
    Vector3d vel;
};

void set_rand_state(Particle *particles) {
    double max_poj_vel = MAX_INIT_VEL / std::sqrt(3);
    std::uniform_real_distribution<double> unif(-max_poj_vel, max_poj_vel);
    std::default_random_engine re;
    re.seed(42);
    Vector3d momentum;
    for (int i = 0; i < NUMBER_PARTICLES; ++i) {
        particles[i].vel.set_values(unif(re), unif(re), unif(re));
        momentum += particles[i].vel;
    }

    for (int i = 0; i < NUMBER_PARTICLES; ++i)
        particles[i].vel -= momentum / NUMBER_PARTICLES;

    const double a = std::ceil(std::cbrt(NUMBER_PARTICLES)); //количество частиц на одно измерение куба
    const double dist = CELL_SIZE / a;

    int i = 0;
    for (double x = -CELL_SIZE / 2 + dist / 2; x < CELL_SIZE / 2; x += dist)
        for (double y = -CELL_SIZE / 2 + dist / 2; y < CELL_SIZE / 2; y += dist)
            for (double z = -CELL_SIZE / 2 + dist / 2; z < CELL_SIZE / 2; z += dist) {
                if (i >= NUMBER_PARTICLES)
                    return;
                particles[i].pos.set_values(x, y, z);
                ++i;
            }
}

Vector3d calc_near_r(const Vector3d &pos1, const Vector3d &pos2) {
    Vector3d res = pos2 - pos1;
    res.x -= round(res.x / CELL_SIZE) * CELL_SIZE;
    res.y -= round(res.y / CELL_SIZE) * CELL_SIZE;
    res.z -= round(res.z / CELL_SIZE) * CELL_SIZE;
    return res;
}

Vector3d calc_force(const Vector3d &pos1, const Vector3d &pos2) {
    static const double max_dist_sqr = FORCE_CUT_DIST * FORCE_CUT_DIST;
    Vector3d r_near = calc_near_r(pos1, pos2);
    double dist_sqr = r_near.sqr();
    if (dist_sqr > max_dist_sqr)
        return Vector3d(0.0);
    else
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
    }
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

std::string path_join(std::initializer_list<std::string> input) {
    std::string res;
    std::string sep = "/";
#ifdef _WIN64
    sep = "\\";
#endif
    for (const auto &s : input)
        res += s + sep;
    res.pop_back();
    return res;
}

void time_test(Particle *particles) {
    std::chrono::high_resolution_clock::time_point start, end;

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5e3; ++i)
        update_state(particles);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Calculation time: " << std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count()
              << "s\n";
    std::cout << particles[0].pos << std::endl;
    std::cout << particles[0].vel << std::endl;
}

void print_data_binary(FILE *out, const Particle *particles, double energy) {
    fwrite(&energy, sizeof(double), 1, out);
    fwrite(particles, sizeof(Particle), NUMBER_PARTICLES, out);
}

void gen_info_file() {
    FILE *outfile = fopen("info.txt", "w");
    fprintf(outfile, "DT                  %e\n", DT);
    fprintf(outfile, "NUMBER_PARTICLES    %d\n", NUMBER_PARTICLES);
    fprintf(outfile, "MAX_INIT_VEL        %e\n", MAX_INIT_VEL);
    fprintf(outfile, "CELL_SIZE           %e\n", CELL_SIZE);
    fprintf(outfile, "MASS                %e\n", MASS);
    fprintf(outfile, "TIME_MODELING       %e\n", TIME_MODELING);
    fprintf(outfile, "TIME_PER_FRAME      %e\n", TIME_PER_FRAME);
    fprintf(outfile, "FORCE_CUT_DIST      %e\n", FORCE_CUT_DIST);
    fclose(outfile);
}

int main() {
    gen_info_file();
    omp_set_num_threads(omp_get_num_procs());

    Particle particles[NUMBER_PARTICLES];
    set_rand_state(particles);

    FILE *out_data_file = fopen("data.bin", "wb");
    fwrite(&NUMBER_PARTICLES, sizeof(int), 1, out_data_file);
    fwrite(&TIME_PER_FRAME, sizeof(double), 1, out_data_file);

    double global_time = 0;

    unsigned n = 0;
    double sum_energies = 0;

    std::chrono::high_resolution_clock::time_point start, end, cur, prev;
    double frac_done, frac_prev;

    start = std::chrono::high_resolution_clock::now();
    while (global_time < TIME_MODELING) {
        frac_done = global_time / TIME_MODELING;
        cur = std::chrono::high_resolution_clock::now();
        double time_left = std::chrono::duration_cast<std::chrono::duration<double>>(cur - prev).count() /
                           (frac_done - frac_prev) *
                           (1 - frac_done);
        frac_prev = frac_done;
        prev = cur;
        double cur_energy = calc_energy_PAR(particles);
        sum_energies += cur_energy;
        n++;
        double mean_energy = sum_energies / n;

        print_data_binary(out_data_file, particles, cur_energy);
        printf("\r                                                            \r");
        printf("%.2f %% (%.0f s left). Energy deviation: %.0e", frac_done * 100, time_left,
               std::abs(1 - cur_energy / mean_energy));
        std::cout.flush();

        double frame_time = 0;
        while (frame_time < TIME_PER_FRAME && global_time < TIME_MODELING) {
            update_state(particles);
            global_time += DT;
            frame_time += DT;
        }
    }
    double cur_energy = calc_energy_PAR(particles);
    print_data_binary(out_data_file, particles, cur_energy);

    end = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "\rCalculation time: " << time_span.count() << "s\n";

    fclose(out_data_file);

    std::string command = "python3 " +
                          path_join({"..", "python_scripts", "make_dirs_and_images.py"}) + " " +
                          ". " +
                          SAVE_PATH;

    int success = system(command.c_str());
    if (success != 0) {
        std::cerr << "Python error";
        exit(1);
    }
}