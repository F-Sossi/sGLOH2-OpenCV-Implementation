//
// Created by Justin on 5/15/2023.
//

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include "ImageMatcher.h"
using namespace cv;

/**
 * Constructor for the ImageMatcher class.
 * Preconditions: The image directory must be valid at "../images/" and contain only jpg images.
 * Postconditions: The imageLibrary vector will be populated with all images in the image directory.
 */
ImageMatcher::ImageMatcher() {
    // Load all images from the image directory in grey scale
    for (const auto & entry : std::filesystem::directory_iterator(image_dir)){
        imageLibrary.push_back(imread(entry.path().string(), IMREAD_GRAYSCALE));
    }
    // Check if images were loaded
    if (imageLibrary.empty()){
        std::cout << "No images were loaded." << std::endl;
    }
}

