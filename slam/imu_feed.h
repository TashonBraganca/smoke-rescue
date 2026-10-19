#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct ImuSample {
    double t;
    std::array<double, 3> a, w;
};

inline std::vector<ImuSample> load_imu(std::istream& input) {
    std::vector<ImuSample> samples;
    std::string line;
    std::size_t number = 0;
    while (std::getline(input, line)) {
        ++number;
        if (line.find_first_not_of(" \t\r") == std::string::npos ||
            line[line.find_first_not_of(" \t\r")] == '#') continue;
        std::istringstream row(line);
        ImuSample s;
        std::string extra;
        if (!(row >> s.t >> s.a[0] >> s.a[1] >> s.a[2] >> s.w[0] >> s.w[1] >> s.w[2]) ||
            (row >> extra))
            throw std::runtime_error("invalid IMU row " + std::to_string(number));
        bool finite = std::isfinite(s.t);
        for (int axis = 0; axis < 3; ++axis)
            finite = finite && std::isfinite(s.a[axis]) && std::isfinite(s.w[axis]);
        if (!finite || (!samples.empty() && s.t <= samples.back().t))
            throw std::runtime_error("IMU timestamps must be finite and strictly increasing");
        samples.push_back(s);
    }
    if (samples.size() < 2) throw std::runtime_error("need at least two IMU samples");
    return samples;
}

// Frame-boundary interpolation avoids extrapolating the last 200 Hz sample to
// the 15 Hz image time. Gravity stays in the accelerometer's specific force.
class ImuFeed {
public:
    ImuFeed(std::vector<ImuSample> samples, double period) : samples_(std::move(samples)) {
        if (samples_.size() < 2 || !std::isfinite(period) || period <= 0)
            throw std::runtime_error("invalid IMU sampling period or sample count");
        for (std::size_t i = 1; i < samples_.size(); ++i) {
            const double dt = samples_[i].t - samples_[i-1].t;
            if (dt <= 0 || dt > 1.5 * period)
                throw std::runtime_error("IMU clock has a gap or non-increasing samples");
        }
    }

    std::vector<ImuSample> between(double start, double end) const {
        if (!(end > start)) throw std::runtime_error("image timestamps must increase");
        std::vector<ImuSample> result{at(start)};
        auto it = std::upper_bound(samples_.begin(), samples_.end(), start,
                                  [](double t, const ImuSample& s) { return t < s.t; });
        for (; it != samples_.end() && it->t < end; ++it) result.push_back(*it);
        result.push_back(at(end));
        return result;
    }

private:
    ImuSample at(double t) const {
        if (!std::isfinite(t) || t < samples_.front().t || t > samples_.back().t)
            throw std::runtime_error("IMU does not bracket image timestamp " + std::to_string(t));
        auto hi = std::lower_bound(samples_.begin(), samples_.end(), t,
                                  [](const ImuSample& s, double time) { return s.t < time; });
        if (hi->t == t) return *hi;
        auto lo = hi - 1;
        const double alpha = (t - lo->t) / (hi->t - lo->t);
        ImuSample s{t, {}, {}};
        for (int axis = 0; axis < 3; ++axis) {
            s.a[axis] = lo->a[axis] + alpha * (hi->a[axis] - lo->a[axis]);
            s.w[axis] = lo->w[axis] + alpha * (hi->w[axis] - lo->w[axis]);
        }
        return s;
    }
    std::vector<ImuSample> samples_;
};
