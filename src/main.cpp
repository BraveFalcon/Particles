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

    LogStream log(path_join({save_path, "log.txt"}));

    FILE *out_data_file = fopen(path_join({save_path, "data.bin"}).c_str(), "wb");
    if (!out_data_file) {
        std::cerr << "Can't create data file" << std::endl;
        exit(1);
    }


    //SystemParticles system_particles(
    //        "/home/brave_falcon/CLionProjects/Particles_git/experiments/4sem/pressure/1.0/data.bin", 0);
    SystemParticles system_particles(NUM_CELLS_PER_DIM, DENSITY, log);
    system_particles.set_vels(5.0, 56);
    system_particles.dt = 1e-3;

    auto start = std::chrono::high_resolution_clock::now();

    log << "Relaxation...\n";
    log.flush();
    int num_iters = 3;
    for (int iter = 0; iter < num_iters; ++iter) {
        system_particles.print_info(1.0 * iter / num_iters);
        system_particles.update_state(ceil(system_particles.get_free_time() * 5 / system_particles.dt));
    }
    log << "Calculation time " << system_particles.print_info(-1) << "\n\n";
    log.flush();

    system_particles.guess_dt(system_particles.get_free_time() * 3);

    //system_particles.npt_berendsen(0.5, 0.9, 0.5);
    system_particles.termostat_berendsen(0.9);

    system_particles.guess_dt(system_particles.get_free_time() * 3);

    log << "NVE...\n";
    log.flush();
    const int ITERS_PER_FRAME = ceil(system_particles.get_free_time() / 10 / system_particles.dt);
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
}
