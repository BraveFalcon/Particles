#pragma once

#include <utility>
#include <cmath>

class Value {
private:
    double sum, sum_sq;
    unsigned n;
public:
    Value() : sum(0.0), sum_sq(0.0), n(0) {}

    void update(double val) {
        sum += val;
        sum_sq += val * val;
        n++;
    }

    [[nodiscard]] std::pair<double, double> get() const {
        double mean = sum / n;
        double mean_sq = sum_sq / n;
        return std::pair<double, double>(mean, sqrt(1.0 * n / (n - 1) * (mean_sq - mean * mean)));
    }

    std::pair<double, double> reset() {
        auto res = get();
        sum = sum_sq = n = 0;
        return res;
    }

    [[nodiscard]] double mean() const { return get().first; }

    [[nodiscard]] double std() const { return get().second; }

    [[nodiscard]] double error_mean() const { return std() / sqrt(n); }

};