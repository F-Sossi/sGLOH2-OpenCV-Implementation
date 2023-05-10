//
// Created by user on 5/10/23.
//

#include "sGLOH2.hpp"
#include <opencv2/imgproc.hpp>

sGLOH2::sGLOH2(int m) : m(m) {}

void sGLOH2::compute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    // Implement method
}

cv::Mat sGLOH2::compute_sGLOH(const cv::Mat& patch) {
    // Implement method
    return cv::Mat();
}

double sGLOH2::distance(const cv::Mat& H_star1, const cv::Mat& H_star2) {
    // Implement method
    return 0.0;
}

