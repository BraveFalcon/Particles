#include "SystemParticles.h"

//TODO::Go on float in all operation, except accumulation
//TODO::Linked list

void SystemParticles::init_arrays() {
    omp_set_num_threads(NUM_THREADS);
    poses = new Vector3d[NUM_PARTICLES];
    prev_forces = new Vector3d[NUM_PARTICLES];
    vels = new Vector3d[NUM_PARTICLES];
    forces = new Vector3d[NUM_PARTICLES];
    th_forces = new Vector3d *[NUM_THREADS];
    th_data = new double *[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; ++i) {
        th_forces[i] = new Vector3d[NUM_PARTICLES];
        th_data[i] = new double[3 * NUM_PARTICLES];
    }
}

SystemParticles::SystemParticles(int num_cells_per_dim, double density) : NUM_THREADS(omp_get_max_threads()),
                                                                          NUM_PARTICLES(4 * num_cells_per_dim *
                                                                                        num_cells_per_dim *
                                                                                        num_cells_per_dim) {
    init_arrays();
    CELL_SIZE = pow(NUM_PARTICLES / density, 1.0 / 3);
    const auto particles_per_dim = static_cast<unsigned >(std::ceil(std::cbrt(NUM_PARTICLES / 4.0)));
    const double dist = CELL_SIZE / particles_per_dim;

    int particle = 0;
    for (int z = 0; z < 2 * particles_per_dim && particle < NUM_PARTICLES; ++z) {
        for (int y = 0; y < 2 * particles_per_dim && particle < NUM_PARTICLES; ++y) {
            for (int x = 0; x < particles_per_dim && particle < NUM_PARTICLES; ++x) {
                poses[particle++] =
                        Vector3d(dist * 0.25) +
                        Vector3d(dist * x + ((z - y) % 2) * dist / 2, dist / 2 * y, dist / 2 * z);
            }
        }
    }
    update_forces();
}

SystemParticles::SystemParticles(std::string file_path, int frame, int num_particles_) : NUM_THREADS(
        omp_get_max_threads()), NUM_PARTICLES(num_particles_) {
    init_arrays();
    FILE *file = fopen(file_path.c_str(), "rb");
    if (!file) {
        fprintf(stderr, "Can't open file in constructor");
        exit(1);
    }
    int num_frames;
    if (!fread(&num_frames, sizeof(int), 1, file)) {
        fprintf(stderr, "Can't read num_frames from bin_file in constructor");
        exit(1);
    }
    if (frame < 0)
        frame = num_frames - frame;
    if (frame >= num_frames || frame < 0) {
        fprintf(stderr, "There isn't frame with this number in constructor");
        exit(1);
    }
    int num_particles;
    if (!fread(&num_particles, sizeof(int), 1, file)) {
        fprintf(stderr, "Can't read num_particles from bin_file in constructor");
        exit(1);
    }
    if (num_particles != NUM_PARTICLES) {
        fprintf(stderr, "Numbers of particles aren't same in constructor");
        exit(1);
    }
    double time_per_frame;
    if (!fread(&time_per_frame, sizeof(double), 1, file)) {
        fprintf(stderr, "Can't read time_per_frame from bin_file in constructor");
        exit(1);
    }
    if (!fread(&CELL_SIZE, sizeof(double), 1, file)) {
        fprintf(stderr, "Can't read cell_size from bin_file in constructor");
        exit(1);
    }
    if (fseek(file, sizeof(double) * (2 + 6 * num_particles) * frame, SEEK_CUR) != 0) {
        fprintf(stderr, "Can't find  %d frame in bin_file in constructor", frame);
        exit(1);
    }

    double energy;
    if (!fread(&energy, sizeof(double), 1, file)) {
        fprintf(stderr, "Can't read energy of %d frame in constructor", frame);
        exit(1);
    }
    double pressure;
    if (!fread(&pressure, sizeof(double), 1, file)) {
        fprintf(stderr, "Can't read pressure of %d frame in constructor", frame);
        exit(1);
    }
    if (fread(poses, sizeof(double), 3 * num_particles, file) != 3 * num_particles) {
        fprintf(stderr, "Can't read poses of %d frame in constructor", frame);
        exit(1);
    }
    if (fread(vels, sizeof(double), 3 * num_particles, file) != 3 * num_particles) {
        fprintf(stderr, "Can't read vels of %d frame in constructor", frame);
        exit(1);
    }
    update_forces();
}

SystemParticles::~SystemParticles() {
    delete[] poses;
    delete[] prev_forces;
    delete[] vels;
    delete[] forces;

    for (int i = 0; i < NUM_THREADS; ++i) {
        delete[] th_forces[i];
        delete[] th_data[i];
    }
    delete[] th_forces;
    delete[] th_data;
}

Vector3d SystemParticles::calc_near_r(const Vector3d &pos1, const Vector3d &pos2) const {
    Vector3d res = pos2 - pos1;
    res.x -= round(res.x / CELL_SIZE) * CELL_SIZE;
    res.y -= round(res.y / CELL_SIZE) * CELL_SIZE;
    res.z -= round(res.z / CELL_SIZE) * CELL_SIZE;
    return res;
}

double SystemParticles::calc_virial(const Vector3d &pos1, const Vector3d &pos2) const {
    Vector3d r_near = calc_near_r(pos1, pos2);
    double dist_sqr = r_near.sqr();
    if (dist_sqr < CUT_DIST * CUT_DIST)
        return 24 * (2 * pow(dist_sqr, -6) - pow(dist_sqr, -3));
    else
        return 0.0;
}

void SystemParticles::update_forces() {
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        for (int th = 0; th < NUM_THREADS; ++th)
            th_forces[th][i].set_values(0.0);
    }
#pragma omp parallel for schedule(dynamic, 1) default(none)
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        int thread = omp_get_thread_num();
        Vector3d pos1 = poses[i];
#pragma omp simd
        for (int j = i + 1; j < NUM_PARTICLES; ++j) {
            th_data[thread][3 * j] = poses[j].x - pos1.x;
            th_data[thread][3 * j + 1] = poses[j].y - pos1.y;
            th_data[thread][3 * j + 2] = poses[j].z - pos1.z;
        }
#pragma omp simd
        for (int j = i + 1; j < NUM_PARTICLES; ++j) {
            th_data[thread][3 * j] -= round(th_data[thread][3 * j] / CELL_SIZE) * CELL_SIZE;
            th_data[thread][3 * j + 1] -= round(th_data[thread][3 * j + 1] / CELL_SIZE) * CELL_SIZE;
            th_data[thread][3 * j + 2] -= round(th_data[thread][3 * j + 2] / CELL_SIZE) * CELL_SIZE;
        }

        for (int j = i + 1; j < NUM_PARTICLES; ++j) {
            double dist_sqr = th_data[thread][3 * j] * th_data[thread][3 * j] +
                              th_data[thread][3 * j + 1] * th_data[thread][3 * j + 1] +
                              th_data[thread][3 * j + 2] * th_data[thread][3 * j + 2];
            if (dist_sqr < CUT_DIST * CUT_DIST) {
                Vector3d force =
                        Vector3d(th_data[thread][3 * j], th_data[thread][3 * j + 1], th_data[thread][3 * j + 2]) *
                        24 * (2 * pow(dist_sqr, -7) - pow(dist_sqr, -4));
                th_forces[thread][i] -= force;
                th_forces[thread][j] += force;
            }
        }
    }
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        prev_forces[i] = forces[i];
        forces[i] = Vector3d(0.0);
        for (int th = 0; th < NUM_THREADS; ++th)
            forces[i] += th_forces[th][i];
    }
}

void SystemParticles::update_state(int num_iters) {
    for (int iter = 0; iter < num_iters; ++iter) {
        for (int i = 0; i < NUM_PARTICLES; ++i)
            poses[i] += vels[i] * dt + 0.5 * forces[i] * dt * dt;
        update_forces();
        for (int i = 0; i < NUM_PARTICLES; ++i)
            vels[i] += 0.5 * (forces[i] + prev_forces[i]) * dt;
    }
}

void SystemParticles::init_bin(FILE *file, int num_frames, double time_per_frame) const {
    fwrite(&num_frames, sizeof(int), 1, file);
    fwrite(&NUM_PARTICLES, sizeof(int), 1, file);
    fwrite(&time_per_frame, sizeof(double), 1, file);
    fwrite(&CELL_SIZE, sizeof(double), 1, file);
}

void SystemParticles::write_bin(FILE *file) const {
    double energy = get_energy();
    double pressure = get_pressure();
    fwrite(&energy, sizeof(double), 1, file);
    fwrite(&pressure, sizeof(double), 1, file);
    fwrite(poses, sizeof(Vector3d), NUM_PARTICLES, file);
    fwrite(vels, sizeof(Vector3d), NUM_PARTICLES, file);
}

void SystemParticles::set_vels(double temperature, unsigned seed) {
    std::normal_distribution<double> distribution(0, std::sqrt(temperature));
    std::default_random_engine random_engine(seed);

    Vector3d sum_vel(0);
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        vels[i] = Vector3d(distribution(random_engine), distribution(random_engine), distribution(random_engine));
        sum_vel += vels[i];
    }

    for (int i = 0; i < NUM_PARTICLES; ++i)
        vels[i] -= sum_vel / NUM_PARTICLES;

}

double SystemParticles::get_temperature() const {
    double res = 0.0;
    for (int i = 0; i < NUM_PARTICLES; ++i)
        res += vels[i].sqr();
    return res / 3 / NUM_PARTICLES;
}

double SystemParticles::get_pressure() const {
    double res = 0;
#pragma omp parallel for reduction(+:res) schedule(dynamic, 1) default(none)
    for (int i = 0; i < NUM_PARTICLES; ++i)
        for (int j = i + 1; j < NUM_PARTICLES; ++j)
            res += calc_virial(poses[i], poses[j]);
    res = NUM_PARTICLES * get_temperature() + res / 3;
    res /= pow(CELL_SIZE, 3);
    return res;
}

double SystemParticles::get_energy() const {
    double energy = 0;
#pragma omp parallel for reduction(+:energy) default(none)
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        energy += 0.5 * vels[i].sqr();
        for (int j = i + 1; j < NUM_PARTICLES; ++j) {
            double dist_sqr = calc_near_r(poses[i], poses[j]).sqr();
            if (dist_sqr < CUT_DIST * CUT_DIST)
                energy +=
                        4 * (pow(dist_sqr, -6) - pow(dist_sqr, -3)) - 4 * (pow(CUT_DIST, -12) - pow(CUT_DIST, -6));
        }
    }
    return energy;
}

void SystemParticles::termostat_andersen(int num_particles, double temp, int seed) {
    std::normal_distribution<double> distribution(0, std::sqrt(temp));
    std::default_random_engine random_engine(seed);

    for (int i = 0; i < num_particles; ++i) {
        int particle = rand() % NUM_PARTICLES;
        vels[particle] = Vector3d(distribution(random_engine), distribution(random_engine),
                                  distribution(random_engine));
    }
}

void SystemParticles::termostat_berendsen(int num_iters, double temp, double tau) {
    for (int iter = 0; iter < num_iters; ++iter) {
        update_state(1);
        double lambda = sqrt(1 + dt / tau * (temp / get_temperature() - 1))
        for (int i = 0; i < NUM_PARTICLES; ++i)
            vels[i] *= lambda;
    }
}