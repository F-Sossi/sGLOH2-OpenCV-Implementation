//
// Created by user on 5/10/23.
//

#ifndef SGLOH_OPENCV_SGLOH2_HPP
#define SGLOH_OPENCV_SGLOH2_HPP

#include <opencv2/core.hpp>

class sGLOH2 {
private:
    int m;  // discretization of the rotation
    cv::Mat H1, H2;
    std::vector<std::vector<cv::Mat>> region_masks;

public:
    explicit sGLOH2(int m = 8);  // default number of bins to 8

    void compute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

    cv::Mat compute_sGLOH(const cv::Mat& patch);

    static double distance(const cv::Mat& H_star1, const cv::Mat& H_star2);

    // Add more methods as needed for other operations, such as matching or orientation estimation
    //cv::Mat computeHistogram(const cv::Mat &region, int m);

    static cv::Mat cyclicShift(const cv::Mat &descriptor, int k);

    cv::Mat compute_sGLOH_single(const cv::Mat &patch);

    cv::Mat computeCustomHistogram(const cv::Mat &data, const std::vector<float> &binEdges);

    cv::Mat computeHistogram(const cv::Mat &patch, const cv::Mat &mask, int m);

    double cosine_similarity(const cv::Mat &H1, const cv::Mat &H2);

    //cv::Mat computeWeightedHistogram(const cv::Mat &orientation, const cv::Mat &magnitude, const cv::Mat &mask, int m);
};

#endif //SGLOH_OPENCV_SGLOH2_HPP
