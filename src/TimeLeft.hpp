#pragma once

#include <string>
#include <chrono>

class TimeLeft {
private:
    double frac_prev = 0, frac_done = 0;
    std::chrono::high_resolution_clock::time_point cur, prev;
    std::chrono::high_resolution_clock::time_point start;
public:
    static std::string secsToTimeStr(double seconds) {
        auto secs = int(seconds);
        auto mins = secs / 60;
        secs %= 60;
        auto hours = mins / 60;
        mins %= 60;
        auto days = hours / 24;
        hours %= 24;
        auto buffer = new char[32];
        if (days > 0)
            sprintf(buffer, "%dd:%dh", days, hours);
        else if (hours > 0)
            sprintf(buffer, "%dh:%dm", hours, mins);
        else if (mins > 0)
            sprintf(buffer, "%dm:%ds", mins, secs);
        else
            sprintf(buffer, "%ds", secs);
        std::string res(buffer);
        delete[] buffer;
        return res;
    }

    TimeLeft() {
        start = std::chrono::high_resolution_clock::now();
    }

    std::string operator()(double fraction_done) {
        frac_done = fraction_done;
        cur = std::chrono::high_resolution_clock::now();
        double time_left = std::chrono::duration_cast<std::chrono::duration<double>>(cur - prev).count() /
                           (frac_done - frac_prev) * (1 - frac_done);
        frac_prev = frac_done;
        prev = cur;

        return secsToTimeStr(time_left);
    }

    std::string get_full_time() {
        auto end = std::chrono::high_resolution_clock::now();
        return secsToTimeStr(std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
    }
};