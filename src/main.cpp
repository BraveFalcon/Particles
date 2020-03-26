#include <iostream>
#include <chrono>
#include <string>
#include "SystemParticles.h"
#include "TimeLeft.hpp"
#include <iomanip>

const int NUM_FRAMES = 100;
//const int ITERS_PER_FRAME = 10;
const int NUM_CELLS_PER_DIM = 7;
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

void print_info(int frame, const SystemParticles &systemParticles) {
    static TimeLeft timeLeft;
    static double init_energy = systemParticles.get_energy();
    double frac_done = 1.0 * frame / NUM_FRAMES;
#ifdef _WIN64
    system("cls");
#else
    system("clear");
#endif
    printf("Complete          %.2f%%\n", frac_done * 100);
    printf("Left              %s\n", timeLeft(frac_done).c_str());
    printf("Energy deviation  %.0e\n", std::abs(1 - systemParticles.get_energy() / init_energy));
    printf("Temperature       %.4f\n", systemParticles.get_temperature());
    printf("Pressure          %.3f\n", systemParticles.get_pressure());
    std::cout.flush();
}

/*
void gen_info_file() {
    FILE *outfile = fopen("info.txt", "w");
    if (!outfile) {
        std::cerr << "Can't create info file" << std::endl;
        exit(1);
    }
    fprintf(outfile, "DT                  %e\n", DT);
    fprintf(outfile, "NUM_PARTICLES    %d\n", NUM_PARTICLES);
    fprintf(outfile, "MAX_INIT_VEL        %e\n", MAX_INIT_VEL);
    fprintf(outfile, "CELL_SIZE           %e\n", CELL_SIZE);
    fprintf(outfile, "TIME_MODELING       %e\n", TIME_MODELING);
    fprintf(outfile, "TIME_PER_FRAME      %e\n", TIME_PER_FRAME);
    fclose(outfile);
}
*/



int main(int argc, char **argv) {
    //gen_info_file();

    if (argc < 2) {
        std::cerr << "You forgot write save path" << std::endl;
        exit(1);
    }
    std::string save_path(argv[1]);


    FILE *out_data_file = fopen(path_join({save_path, "data.bin"}).c_str(), "wb");
    if (!out_data_file) {
        std::cerr << "Can't create data file" << std::endl;
        exit(1);
    }


    //SystemParticles system_particles(
    //        "/home/brave_falcon/CLionProjects/Particles_git/experiments/4sem/pressure/1.0/data.bin", 0);
    SystemParticles system_particles(NUM_CELLS_PER_DIM, DENSITY);
    system_particles.set_vels(1.0, 56);

    printf("Preparing... (%f)\n", system_particles.get_free_time());
    system_particles.update_state(ceil(system_particles.get_free_time() * 5 / system_particles.dt));

    printf("NPT... (%f)\n", system_particles.get_free_time());
    system_particles.guess_dt(system_particles.get_free_time() * 2);
    system_particles.npt_berendsen(0.5, 1, system_particles.get_free_time() * 4, 1);

    printf("NVE... (%f)\n", system_particles.get_free_time());
    system_particles.guess_dt(system_particles.get_free_time() * 2);
    const int ITERS_PER_FRAME = ceil(system_particles.get_free_time() / 10 / system_particles.dt);
    system_particles.init_bin(out_data_file, NUM_FRAMES, system_particles.dt * ITERS_PER_FRAME);

    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    for (int frame = 0; frame < NUM_FRAMES; ++frame) {
        print_info(frame, system_particles);
        system_particles.write_bin(out_data_file);
        system_particles.update_state(ITERS_PER_FRAME);
    }
    print_info(NUM_FRAMES, system_particles);
    system_particles.write_bin(out_data_file);
    end = std::chrono::high_resolution_clock::now();
    auto calc_time = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    printf("%.3e", calc_time);
    std::cout << "\nCalculation time  " << TimeLeft::secsToTimeStr(calc_time) << "\n\n";

    fclose(out_data_file);
    std::string command =
            "python3 " + path_join({path_get_head(argv[0]), "..", "python_scripts", "make_dirs_and_images.py"}) +
            " " +
            save_path + " " + save_path;

    if (std::system(command.c_str())) {
        std::cerr << "Python error";
        exit(1);
    }

}
