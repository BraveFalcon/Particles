#include <iostream>
#include <chrono>
#include <string>
#include "../model_constants.h"
#include <omp.h>
#include "SystemParticles.h"


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


    FILE *out_data_file = fopen("data.bin", "wb");
    fwrite(&NUMBER_PARTICLES, sizeof(int), 1, out_data_file);
    fwrite(&TIME_PER_FRAME, sizeof(double), 1, out_data_file);

    SystemParticles system_particles(42);
    double init_energy = system_particles.get_energy();

    std::chrono::high_resolution_clock::time_point start, end, cur, prev;
    double frac_done, frac_prev;

    start = std::chrono::high_resolution_clock::now();
    while (system_particles.get_model_time() < TIME_MODELING - DT) {
        frac_done = system_particles.get_model_time() / TIME_MODELING;
        cur = std::chrono::high_resolution_clock::now();
        double time_left = std::chrono::duration_cast<std::chrono::duration<double>>(cur - prev).count() /
                           (frac_done - frac_prev) *
                           (1 - frac_done);
        frac_prev = frac_done;
        prev = cur;

        system_particles.write_bin(out_data_file);
        printf("\r                                                            \r");
        printf("%.2f %% (%.0f s left). Energy deviation: %.0e", frac_done * 100, time_left,
               std::abs(1 - system_particles.get_energy() / init_energy));
        std::cout.flush();
        system_particles.update_state(TIME_PER_FRAME);
    }
    system_particles.write_bin(out_data_file);

    end = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "\rCalculation time: " << time_span.count() << "s\n";

    fclose(out_data_file);

    std::string command = "python3 " +
                          path_join({"..", "python_scripts", "make_dirs_and_images.py"}) + " " +
                          ". " +
                          SAVE_PATH;

    int success = std::system(command.c_str());
    if (success != 0) {
        std::cerr << "Python error";
        exit(1);
    }
}