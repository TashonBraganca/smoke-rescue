"""Compile and execute the same timestamp batching code used by the container."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_imu_interpolates_frame_boundaries_and_rejects_bad_clock(tmp_path):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable')
    source = tmp_path / 'test.cc'
    source.write_text(r'''
#include "imu_feed.h"
#include <cassert>
#include <sstream>
int main() {
    std::istringstream input("# t ax ay az gx gy gz\n0.005 0 -9.81 1 0 1 0\n0.010 0 -9.81 2 0 2 0\n0.015 0 -9.81 3 0 3 0\n0.020 0 -9.81 4 0 4 0\n0.025 0 -9.81 5 0 5 0\n");
    const auto samples = load_imu(input);
    ImuFeed feed(samples, .005);
    // 15 Hz images need samples on the image timestamps, not rounded IMU ticks.
    auto first = feed.between(.006, .013);
    assert(first.size() == 3);
    assert(first.front().t == .006 && first.back().t == .013);
    assert(std::abs(first.front().a[2] - 1.2) < 1e-9);
    assert(std::abs(first.back().w[1] - 2.6) < 1e-9);
    auto second = feed.between(.013, .021);
    assert(second.front().t == .013 && second.back().t == .021);
    assert(second.size() == 4);
    assert(second[1].t == .015 && second[2].t == .020);
    bool rejected = false;
    try { feed.between(.001, .013); } catch(const std::exception&) { rejected = true; }
    assert(rejected);
    rejected = false;
    try { feed.between(.02, .03); } catch(const std::exception&) { rejected = true; }
    assert(rejected);
    for (auto text : {"0.01 0 0 0 0 0 0\n0.005 0 0 0 0 0 0\n",
                      "0.01 0 0 0 0 0\n", "0.01 0 0 0 0 0 0 extra\n"}) {
        rejected = false;
        try { std::istringstream bad(text); load_imu(bad); }
        catch(const std::exception&) { rejected = true; }
        assert(rejected);
    }
    rejected = false;
    try { ImuFeed gap({samples.front(), samples.back()}, .005); }
    catch(const std::exception&) { rejected = true; }
    assert(rejected);
}
''')
    root = Path(__file__).parents[2]
    exe = tmp_path / 'imu_test'
    subprocess.run([compiler, '-std=c++17', '-I', str(root / 'slam'), str(source), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
