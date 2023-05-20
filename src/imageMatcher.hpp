//
// Created by Justin on 5/15/2023 hu.
//
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#ifndef SGLOH_OPENCV_IMAGESEARCH_HPP
#define SGLOH_OPENCV_IMAGESEARCH_HPP
/**
 * @brief This class is used to compare images using the SIFT and sGLOH2 algorithms.
 */
class ImageMatcher {
public:
    ImageMatcher();
    std::vector<cv::Mat> siftMatch(const cv::Mat& image);
    std::vector<cv::Mat> sGLOHMatch(const cv::Mat &image, int m);
private:
    std::string image_dir = "../images/";
    std::vector<cv::Mat> imageLibrary;
    cv::Ptr<cv::SIFT> siftDetector = cv::SIFT::create();
    std::vector<std::vector<cv::KeyPoint>> siftKeypoints; // A little unwieldy, but now we only have to compute keypoints once
    std::vector<cv::DMatch> scoreMatches(const cv::Mat &queryDescriptors, const cv::Mat &libraryDescriptors, std::vector<cv::Mat> &matchingImages, int imageIndex);
};

#endif //SGLOH_OPENCV_IMAGESEARCH_HPP

