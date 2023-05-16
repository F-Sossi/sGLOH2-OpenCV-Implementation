//
// Created by Justin on 5/15/2023.
//

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include "imageMatcher.hpp"
using namespace cv;


const double MIN_DIST = 100;
const float RATIO_THRESHOLD = 0.03; //Optimize this
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

/**
 *
 * @param image to be compared to the image library
 * @return a vector of the best matches
 * Preconditions:The image must be a valid jpg image and the image library must be populated.
 * Postconditions: The vector will contain the best matches from the image library determined using SIFT detectors.
 */
std::vector<cv::Mat> ImageMatcher::siftMatch(const cv::Mat &image) {
    Ptr<SIFT> detector = SIFT::create();
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    detector->detectAndCompute(image, noArray(), keypoints, descriptors);
    std::vector<cv::Mat> bestMatches;
    for (int i = 0; i < imageLibrary.size(); i++){ // for each image in the library
        // Display progress
        std::cout << "Processing image " << i+1 << " of " << imageLibrary.size() << std::endl;

        // Compute keypoints and descriptors for the library image
        std::vector<KeyPoint> libraryKeypoints; // Keypoints for the library image
        Mat libraryDescriptors;
        detector->detectAndCompute(imageLibrary[i], noArray(), libraryKeypoints, libraryDescriptors);

        // Match the descriptors
        BFMatcher matcher(NORM_L2); //Optimize this
        std::vector<DMatch> matches;
        matcher.match(descriptors, libraryDescriptors, matches);  //Optimize this (matcher.knnMatch(descriptors1, descriptors2, matches, 2) ?

        // Score the matches to calculate an overall match score for the library image to the input image.
        std::sort(matches.begin(), matches.end());
        std::vector<DMatch> goodMatches;


        // Iterates through the matches and filters out low quality matches
        for( int i = 0; i < descriptors.rows; i++ ) {
            if( matches[i].distance <= std::max(2*MIN_DIST, 0.02) ) {  //Optimize this
                goodMatches.push_back( matches[i]);
            }
        }
        //Mat img_matches;
        //drawMatches(image, keypoints, imageLibrary[i], libraryKeypoints, goodMatches, img_matches);
        // If there are enough good matches, add the library image to the best matches vector
        if(matches.size() == 0){
            std::cout << "No matches found for image " << i+1 << std::endl;
        }
        else if((float(goodMatches.size())/float(matches.size())) >= RATIO_THRESHOLD){ //Optimize this
            bestMatches.push_back(imageLibrary[i]);
        }
        else{
            std::cout << "Error finding matches for image " << i+1 << std::endl;
        }
    }
    return bestMatches;
}

