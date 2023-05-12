//
// Created by user on 5/10/23.
//

#include "sGLOH2.hpp"
#include <opencv2/xfeatures2d.hpp>

sGLOH2::sGLOH2(int m) : m(m) {
    // Initialize any other member variables as needed
}

void sGLOH2::compute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    // Create a SIFT detector
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();

    // Detect keypoints
    detector->detect(image, keypoints);

    // Compute descriptors for each keypoint
    for (auto& keypoint : keypoints) {
        cv::Mat patch;
        // Extract patch around keypoint
        // You might need to use cv::getRectSubPix or similar function
        // Make sure to handle edge cases where the keypoint is near the border of the image

        // Compute sGLOH descriptor for the patch
        cv::Mat descriptor = compute_sGLOH(patch);

        // Append descriptor to descriptors matrix
        descriptors.push_back(descriptor);
    }
}

cv::Mat sGLOH2::compute_sGLOH(const cv::Mat& patch) {
    // Implement your sGLOH computation here
    // This will likely involve computing the gradient orientation histogram for the patch,
    // and then applying the spatial and orientation weighting functions
}

double sGLOH2::distance(const cv::Mat& H_star1, const cv::Mat& H_star2) {
    // Implement your distance computation here
    // This will likely involve computing the chi-squared distance between the two histograms
}


