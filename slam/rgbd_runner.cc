/**
 * Headless RGB-D runner derived from ORB-SLAM3's rgbd_tum example.
 *
 * ORB-SLAM3 is GPLv3. This runner is intended to be compiled and distributed
 * with ORB-SLAM3 under the same license terms.
 */

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <System.h>

namespace fs = std::filesystem;

struct AssociatedFrame {
    double rgb_timestamp;
    fs::path rgb_path;
    double depth_timestamp;
    fs::path depth_path;
};

std::vector<AssociatedFrame> load_associations(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open associations file: " + path.string());
    }

    std::vector<AssociatedFrame> frames;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line.front() == '#') {
            continue;
        }

        std::istringstream row(line);
        AssociatedFrame frame;
        if (!(row >> frame.rgb_timestamp >> frame.rgb_path >> frame.depth_timestamp >> frame.depth_path)) {
            throw std::runtime_error(
                "invalid association row " + std::to_string(line_number) + ": " + line
            );
        }
        frames.push_back(std::move(frame));
    }

    if (frames.empty()) {
        throw std::runtime_error("associations file contains no frames: " + path.string());
    }
    return frames;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: smoke_rescue_rgbd ROBOT_FOLDER OUTPUT_FOLDER\n";
        return 2;
    }

    try {
        const fs::path robot_folder = fs::path(argv[1]);
        const fs::path output_folder = fs::path(argv[2]);
        const fs::path settings = robot_folder / "camera.yaml";
        const fs::path associations = robot_folder / "associations.txt";
        const fs::path vocabulary = "/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt";

        if (!fs::is_regular_file(settings)) {
            throw std::runtime_error("missing camera settings: " + settings.string());
        }
        if (!fs::is_regular_file(vocabulary)) {
            throw std::runtime_error("missing ORB vocabulary: " + vocabulary.string());
        }
        fs::create_directories(output_folder);

        cv::FileStorage camera_settings(settings.string(), cv::FileStorage::READ);
        const bool expects_rgb = static_cast<int>(camera_settings["Camera.RGB"]) != 0;

        const std::vector<AssociatedFrame> frames = load_associations(associations);
        ORB_SLAM3::System slam(
            vocabulary.string(),
            settings.string(),
            ORB_SLAM3::System::RGBD,
            false
        );
        const float image_scale = slam.GetImageScale();

        std::cout << "Processing " << frames.size() << " RGB-D frames without a viewer\n";
        for (std::size_t index = 0; index < frames.size(); ++index) {
            cv::Mat rgb = cv::imread(
                (robot_folder / frames[index].rgb_path).string(),
                cv::IMREAD_UNCHANGED
            );
            cv::Mat depth = cv::imread(
                (robot_folder / frames[index].depth_path).string(),
                cv::IMREAD_UNCHANGED
            );
            if (rgb.empty() || depth.empty()) {
                throw std::runtime_error(
                    "failed to load RGB-D pair at timestamp "
                    + std::to_string(frames[index].rgb_timestamp)
                );
            }
            // imread decodes BGR, whereas Camera.RGB selects the input order.
            if (expects_rgb && rgb.channels() == 3) {
                cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
            } else if (expects_rgb && rgb.channels() == 4) {
                cv::cvtColor(rgb, rgb, cv::COLOR_BGRA2RGBA);
            }

            if (image_scale != 1.0F) {
                const cv::Size size(
                    static_cast<int>(rgb.cols * image_scale),
                    static_cast<int>(rgb.rows * image_scale)
                );
                cv::resize(rgb, rgb, size);
                cv::resize(depth, depth, size, 0, 0, cv::INTER_NEAREST);
            }

            const auto started = std::chrono::steady_clock::now();
            slam.TrackRGBD(rgb, depth, frames[index].rgb_timestamp);
            const auto elapsed = std::chrono::steady_clock::now() - started;

            double frame_period = 0.0;
            if (index + 1 < frames.size()) {
                frame_period = frames[index + 1].rgb_timestamp - frames[index].rgb_timestamp;
            } else if (index > 0) {
                frame_period = frames[index].rgb_timestamp - frames[index - 1].rgb_timestamp;
            }
            const auto budget = std::chrono::duration<double>(std::max(0.0, frame_period));
            if (elapsed < budget) {
                std::this_thread::sleep_for(budget - elapsed);
            }
        }

        slam.Shutdown();
        slam.SaveTrajectoryTUM((output_folder / "CameraTrajectory.txt").string());
        slam.SaveKeyFrameTrajectoryTUM(
            (output_folder / "KeyFrameTrajectory.txt").string()
        );
        std::cout << "Wrote trajectories to " << output_folder << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "smoke_rescue_rgbd: " << error.what() << '\n';
        return 1;
    }
}
