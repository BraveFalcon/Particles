#include "SystemParticles.h"

//TODO::Go on float in all operation, except accumulation
//TODO::Linked list
//TODO::Go to sdtl vectors


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

SystemParticles::SystemParticles(int num_cells_per_dim, double density) : NUM_PARTICLES(4 * num_cells_per_dim *
                                                                                        num_cells_per_dim *
                                                                                        num_cells_per_dim) {
    init_arrays();
    cell_size = pow(NUM_PARTICLES / density, 1.0 / 3);
    const auto particles_per_dim = static_cast<unsigned >(std::ceil(std::cbrt(NUM_PARTICLES / 4.0)));
    const double dist = cell_size / particles_per_dim;

    int particle = 0;
    for (int z = 0; z < 2 * particles_per_dim && particle < NUM_PARTICLES; ++z) {
        for (int y = 0; y < 2 * particles_per_dim && particle < NUM_PARTICLES; ++y) {
            for (int x = 0; x < particles_per_dim && particle < NUM_PARTICLES; ++x) {
                poses[particle++] =
                        Vector3d(dist * 0.25) - Vector3d(cell_size / 2) +
                        Vector3d(dist * x + (abs(z - y) % 2) * dist / 2, dist / 2 * y, dist / 2 * z);
            }
        }
    }
    update_forces();
}

SystemParticles::SystemParticles(std::string file_path, int frame, int num_particles_) : NUM_PARTICLES(num_particles_) {
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
    if (!fread(&cell_size, sizeof(double), 1, file)) {
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

Vector3d SystemParticles::get_near_r(const Vector3d &pos1, const Vector3d &pos2) const {
    Vector3d res = pos2 - pos1;
    res.x -= round(res.x / cell_size) * cell_size;
    res.y -= round(res.y / cell_size) * cell_size;
    res.z -= round(res.z / cell_size) * cell_size;
    return res;
}

void SystemParticles::update_forces() {
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        for (int th = 0; th < NUM_THREADS; ++th)
            th_forces[th][i].set_values(0.0);
    }
    double _pot_energy = 0;
    double _virial = 0;
#pragma omp parallel for schedule(dynamic, 1) default(none) reduction(+:_pot_energy, _virial)
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
            th_data[thread][3 * j] -= round(th_data[thread][3 * j] / cell_size) * cell_size;
            th_data[thread][3 * j + 1] -= round(th_data[thread][3 * j + 1] / cell_size) * cell_size;
            th_data[thread][3 * j + 2] -= round(th_data[thread][3 * j + 2] / cell_size) * cell_size;
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
                _pot_energy +=
                        4 * (pow(dist_sqr, -6) - pow(dist_sqr, -3)) - 4 * (pow(CUT_DIST, -12) - pow(CUT_DIST, -6));
                _virial += 24 * (2 * pow(dist_sqr, -6) - pow(dist_sqr, -3));
            }
        }
    }
    pot_energy = _pot_energy;
    virial = _virial;
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

void SystemParticles::init_bin_file(FILE *file, int num_frames, double time_per_frame) const {
    fwrite(&num_frames, sizeof(int), 1, file);
    fwrite(&NUM_PARTICLES, sizeof(int), 1, file);
    fwrite(&time_per_frame, sizeof(double), 1, file);
    fwrite(&cell_size, sizeof(double), 1, file);
}

void SystemParticles::write_bin_file(FILE *file) const {
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
    return (NUM_PARTICLES * get_temperature() + virial / 3) / pow(cell_size, 3);
}

double SystemParticles::get_energy() const {
    return pot_energy + 1.5 * get_temperature() * NUM_PARTICLES;
}

double SystemParticles::get_free_time() const {
    double temp = get_temperature();
    double sigma = M_PI * pow(2.0 / 3.0 / temp * (sqrt(1 + 3 * temp) - 1), 1.0 / 3);
    return pow(cell_size, 3) / sqrt(6 * temp) / NUM_PARTICLES / sigma;
}

double SystemParticles::calc_rel_std_energy(double DT, double time) {
    double old_dt = dt;
    dt = DT;
    Value energy;
    int iters = 50;
    for (int i = 0; i < iters; ++i) {
        update_state(ceil(time / dt / iters));
        energy.update(get_energy());
    }
    dt = old_dt;
    double fluct = energy.std() / std::abs(energy.mean());
    printf("%-9.0e%.1e\n", DT, fluct);
    return fluct;
}

void SystemParticles::guess_dt(double iter_time) {
    printf("Selecting dt... (free time: %f)\n", get_free_time());
    printf("DT       Fluctuation\n");
    double min_fluct = 1e-5;
    double max_fluct = 5e-5;
    double mean_fluct = 0.5 * (min_fluct + max_fluct);
    double dt_l, dt_r;
    double fluct_l, fluct_r;
    dt_l = dt;
    fluct_l = calc_rel_std_energy(dt_l, iter_time);
    if (fluct_l > max_fluct) {
        dt_r = dt_l;
        fluct_r = fluct_l;
        dt_l = dt_r / 5;
        fluct_l = calc_rel_std_energy(dt_l, iter_time);
    } else if (fluct_l < min_fluct) {
        dt_r = dt_l * 5;
        fluct_r = calc_rel_std_energy(dt_r, iter_time);
    } else {
        printf("\n");
        return;
    }
    if (fluct_r < max_fluct && fluct_r > min_fluct) {
        dt = dt_r;
        printf("\n");
        return;
    }
    double dt_m, fluct_m;
    do {
        dt_m = dt_l + (dt_r - dt_l) * (mean_fluct - fluct_l) / (fluct_r - fluct_l);
        if (dt_m <= 0) dt_m = dt_l / 2;
        fluct_m = calc_rel_std_energy(dt_m, iter_time);
        if (dt_m > dt_r) {
            dt_l = dt_r;
            fluct_l = fluct_r;
            dt_r = dt_m;
            fluct_r = fluct_m;
        } else if (dt_m < dt_l) {
            dt_r = dt_l;
            fluct_r = fluct_l;
            dt_l = dt_m;
            fluct_l = fluct_m;
        } else if (fluct_m > mean_fluct) {
            dt_r = dt_m;
            fluct_r = fluct_m;
        } else {
            dt_l = dt_m;
            fluct_l = dt_m;
        }
    } while (fluct_m < min_fluct || fluct_m > max_fluct);
    dt = dt_m;
    printf("\n");
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
        double lambda = sqrt(1 + dt / tau * (temp / get_temperature() - 1));
        for (int i = 0; i < NUM_PARTICLES; ++i)
            vels[i] *= lambda;
    }
}

void SystemParticles::npt_berendsen(unsigned num_iters, double press, double temp, double tau, double beta) {
    double min_mu = 0.8, max_mu = 1.25;
    for (unsigned iter = 0; iter < num_iters; ++iter) {
        update_state(1);

        if (get_temperature() > 10) {
            fprintf(stderr, "Heat boom in berendsen\n");
            set_vels(temp);
        }

        double lambda = sqrt(1 + 2 * dt / tau * (temp / get_temperature() - 1));
        double mu = pow(1 - beta * dt / tau * (press - get_pressure()), 1.0 / 3);

        if (mu < min_mu) mu = min_mu;
        else if (mu > max_mu) mu = max_mu;

        for (int i = 0; i < NUM_PARTICLES; ++i) {
            poses[i] = get_near_r(Vector3d(0.0), poses[i]);
            poses[i] *= mu;
            vels[i] *= lambda;
        }
        cell_size *= mu;
        update_forces();
    }
}

void SystemParticles::npt_berendsen(double press, double temp, double mean_beta) {
    class Beta {
    private:
        double prev_vol, prev_press;
        double beta;
        const double min_beta = 1e-4, max_beta = 1e4;
        const double init_beta;
    public:
        Beta(double cellSize, double pressure, double mean_beta) : init_beta(mean_beta) {
            prev_vol = pow(cellSize, 3);
            prev_press = pressure;
            beta = init_beta;
        }

        void update(double cellSize, double pressure) {
            double volume = pow(cellSize, 3);
            beta = -(volume - prev_vol) / (pressure - prev_press) * 2 / (volume + prev_vol);
            if (beta < 0) beta = init_beta;
            else if (beta < min_beta) beta = min_beta;
            else if (beta > max_beta) beta = max_beta;

            prev_vol = volume;
            prev_press = pressure;
        }

        double operator*(double x) { return beta * x; };
    };
    printf("NPT...\n");
    print_info(0.0);
    Beta beta(cell_size, get_pressure(), mean_beta);
    Value cur_temp, cur_press;
    double init_tau = get_free_time();
    //FILE *outfile = fopen("/home/brave_falcon/CLionProjects/Particles_git/experiments/new/params.csv", "w");
    double prev_press, prev_temp;
    do {
        prev_press = cur_press.mean();
        prev_temp = cur_temp.mean();
        cur_temp.reset();
        cur_press.reset();
        double tau = get_free_time();
        if (std::abs(tau - init_tau) / init_tau > 0.5) {
            printf("\n");
            guess_dt(tau * 3);
            init_tau = tau;
        }
        int num_iters = 16;
        for (int iter = 0; iter < num_iters; ++iter) {
            npt_berendsen(ceil(tau / dt / num_iters), press, temp, tau, beta * 1.0);
            cur_temp.update(get_temperature());
            cur_press.update(get_pressure());
        }
        beta.update(cell_size, get_pressure());
        //printf("  %.2f\t%.5f\t%.5f\n", beta * 1.0, cur_temp.error_mean(), cur_press.error_mean());
        print_info(0.0);
        //fprintf(outfile, "%f\t%f\t%f\t%f\n", cur_temp.mean(), cur_temp.error_mean(), std::abs(temp - cur_temp.mean()) / cur_temp.error_mean(),
        //std::abs(prev_temp - cur_temp.mean()) / cur_temp.error_mean());
    } while (std::abs(press - cur_press.mean()) > cur_press.error_mean()
             || std::abs(prev_press - cur_press.mean()) > cur_press.error_mean()
             || std::abs(temp - cur_temp.mean()) > cur_temp.error_mean()
             || std::abs(prev_temp - cur_temp.mean()) > cur_temp.error_mean()
            );
    //fclose(outfile);
    std::cout << "Calculation time " << print_info(-1) << "\n\n";


}

//TODO::запись лог файла примерно как в lammps
std::string SystemParticles::print_info(double frac_done) const {
    static TimeLeft timeLeft;
    static double init_energy = 0;
    static int print_info_iter = 0;
    if (frac_done < 0) {
        std::string full_time = timeLeft.get_full_time();
        timeLeft = TimeLeft();
        init_energy = 0;
        print_info_iter = 0;
        return full_time;
    }
    if (init_energy == 0) init_energy = get_energy();
    if (print_info_iter == 0 || print_info_iter > 9) {
        printf("%%   Left     E_dev      Temp     Press    Dens     Free_time\n");
        print_info_iter = 0;
    }
    printf("%-4.0f%-9s%.1e    %.3f    %-9.3f%-9.3f%.3f\n", frac_done * 100, timeLeft(frac_done).c_str(),
           std::abs(1 - get_energy() / init_energy), get_temperature(), get_pressure(),
           NUM_PARTICLES / pow(cell_size, 3), get_free_time());
    //std::cout.flush();
    print_info_iter++;
    return timeLeft.get_full_time();
}
