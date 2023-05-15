//
// Created by Justin on 5/15/2023.
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
private:
    std::string image_dir = "../images/";
    std::vector<cv::Mat> imageLibrary;
};

#endif SGLOH_OPENCV_IMAGESEARCH_HPP

