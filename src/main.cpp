#include <iostream>
#include <chrono>
#include <string>
#include "SystemParticles.h"
#include <iomanip>
#include "LogStream.h"


const int NUM_FRAMES = 100;
const int NUM_CELLS_PER_DIM = 5;
const double DENSITY = 0.9;

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

std::string path_get_head(const std::string &path) {
    std::string sep = "/";
#ifdef _WIN64
    sep = "\\";
#endif
    size_t pos = path.find_last_of(sep);
    if (pos == std::string::npos)
        return "";
    return path.substr(0, pos);
}

const std::vector<std::pair<double, double>> dots = {
        {1.08,    3.0},
        {1.06,    3.0},
        {1.04,    3.0},
        {1.02,    3.0},
        {1.0,     3.0},
        {1.0,     2.262},
        {1.0,     1.525},
        {1.0348,  1.525},
        {1.0698,  1.525},
        {1.10476, 1.525},
        {1.14,    1.525},
        {1.2143,  0.9016},
        {1.2143,  0.6894},
        {1.2143,  0.4772},
        {1.2143,  0.2640},
        {1.0,     0.7876},
        {1.0,     0.05},
        {1.10657, 0.05},
        {1.2143,  0.05},
        {1.32083, 0.05},
        {1.42857, 0.05},
        {1.42857, 0.1328},
        {1.42857, 0.2152},
        {1.42857, 0.2981},
        {1.42857, 0.3806},
};

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "You forgot write save path" << std::endl;
        exit(1);
    }
    std::string save_path(argv[1]);

    LogStream log;

    SystemParticles system_particles(NUM_CELLS_PER_DIM, DENSITY, log);
    system_particles.set_vels(1.9, 56);
    system_particles.dt = 5e-3;

    log << "Relaxation...\n";
    log.flush();
    int num_iters = 3;
    for (int iter = 0; iter < num_iters; ++iter) {
        system_particles.print_info(1.0 * iter / num_iters);
        system_particles.update_state(ceil(system_particles.get_free_time() * 5 / system_particles.dt));
    }
    log << "Calculation time " << system_particles.print_info(-1) << "\n\n";
    log.flush();

    for (auto dot : dots) {
        FILE *out_data_file = fopen(path_join({save_path, "data.bin"}).c_str(), "wb");
        if (!out_data_file) {
            std::cerr << "Can't create data file" << std::endl;
            exit(1);
        }
        system_particles.init_log_file(path_join({save_path, "log.txt"}));

        log << "Goals: \n";
        log << "Press " << std::fixed << std::setprecision(4) << dot.second << '\n';
        log << "Temp " << std::fixed << std::setprecision(4) << 1.0 / dot.first << "\n\n";

        auto start = std::chrono::high_resolution_clock::now();

        system_particles.guess_dt(system_particles.get_free_time() * 3);

        system_particles.npt_berendsen(dot.second, 1.0 / dot.first);

        system_particles.guess_dt(system_particles.get_free_time() * 3);

        log << "NVE...\n";
        log.flush();
        const int ITERS_PER_FRAME = ceil(system_particles.get_free_time() / 20 / system_particles.dt);
        system_particles.init_bin_file(out_data_file, NUM_FRAMES, system_particles.dt * ITERS_PER_FRAME);

        for (int frame = 0; frame < NUM_FRAMES; ++frame) {
            system_particles.print_info(1.0 * frame / NUM_FRAMES);
            system_particles.write_bin_file(out_data_file);
            system_particles.update_state(ITERS_PER_FRAME);
        }

        log << "Calculation time  " << system_particles.print_info(-1) << "\n\n";
        log.flush();
        fclose(out_data_file);

        auto end = std::chrono::high_resolution_clock::now();
        log << "Full calculation time  "
            << TimeLeft::secsToTimeStr(std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count())
            << '\n';
        log.flush();
        start = std::chrono::high_resolution_clock::now();
        std::string command =
                "python3 " + path_join({path_get_head(argv[0]), "..", "python_scripts", "make_dirs_and_images.py"}) +
                " " +
                save_path + " " + save_path;

        if (std::system(command.c_str())) {
            log << "Python error\n";
            log.flush();
            exit(1);
        }
        end = std::chrono::high_resolution_clock::now();
        log << "Python time  "
            << TimeLeft::secsToTimeStr(std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count())
            << '\n';
        log.close();
    }
}
