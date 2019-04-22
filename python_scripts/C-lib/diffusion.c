void get_mean_sqrs_delta_r(
        int num_frames,
        int num_particles,
        const double *data,
        double *res) {
    int sizeof_particle = 6 * sizeof(double);
    int sizeof_frame = num_particles * sizeof_particle;
    res[0] = 0.0;
    for (int tau = 1; tau < num_frames; ++tau) {
        res[tau] = 0.0;
        for (int start_point = 0; start_point < num_frames - tau; start_point += tau)
            for (int particle = 0; particle < num_particles; ++particle) {
                const double *pos1 = (void *) (data) + start_point * sizeof_frame + particle * sizeof_particle;
                const double *pos2 = (void *) (data) + (start_point + tau) * sizeof_frame + particle * sizeof_particle;
                for (int axis = 0; axis < 3; ++axis) {
                    double proj = pos2[axis] - pos1[axis];
                    res[tau] += proj * proj;
                }
            }
        res[tau] /= num_particles * ((num_frames - 1) / tau);
    }
}
