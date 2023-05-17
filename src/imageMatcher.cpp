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
#include "sGLOH2.hpp"
using namespace cv;


const double MIN_DIST = 100;
const float RATIO_THRESHOLD = 0.1; //Optimize this
/**
 * Constructor for the ImageMatcher class.
 * Preconditions: The image directory must be valid at "../images/" and contain only jpg images.
 * Postconditions: The imageLibrary vector will be populated with all images in the image directory.
 */
ImageMatcher::ImageMatcher() {
    // Load all images from the image directory in grey scale
    // If it is not a .jpg ignore it
    for (const auto & entry : std::filesystem::directory_iterator(image_dir)){
        if (entry.path().extension() == ".jpg"){
            imageLibrary.push_back(imread(entry.path().string(), IMREAD_GRAYSCALE));
        }
    }
    // Check if images were loaded
    if (imageLibrary.empty()){
        std::cout << "No images were loaded." << std::endl;
    }
    siftDetector->detect(imageLibrary, siftKeypoints);
}

/**
 * @param queryDescriptors from the input image
 * @param libraryDescriptors from the library image to check for match
 * @param matchingImages vector of the images that pass the match threshold
 * Preconditions: The descriptors have been calculated for both images.
 * Postconditions: If the library image passes the match threshold, it will be added to the goodMatches vector.
 */
std::vector<DMatch> ImageMatcher::scoreMatches(const Mat &queryDescriptors, const Mat &libraryDescriptors, std::vector<cv::Mat> &matchingImages, int imageIndex) {
    // Match the descriptors
    BFMatcher matcher(NORM_L2); //Optimize this
    std::vector<DMatch> matches;
    matcher.match(queryDescriptors, libraryDescriptors,
                  matches);  //Optimize this (matcher.knnMatch(descriptors1, descriptors2, matches, 2) ?

    // Iterates through the matches and filters out low quality matches
    std::vector<DMatch> goodMatches;
    for (int i = 0; i < queryDescriptors.rows; i++) {
        if (matches[i].distance <= std::max(2 * MIN_DIST, 0.02)) {  //Optimize this
            goodMatches.push_back(matches[i]);
        }
    }
    // If there are enough good matches, add the library image to the best matches vector
    if (matches.size() == 0) {
        std::cout << "No matches found for image " << imageIndex + 1 << std::endl;
    } else if ((float(goodMatches.size()) / float(matches.size())) >= RATIO_THRESHOLD) { //Optimize this
        matchingImages.push_back(imageLibrary[imageIndex]);
        std::cout << "Image " << imageIndex + 1 << " passed the match threshold." << std::endl;
    } else {
        std::cout << "Image " << imageIndex + 1 << " failed the match threshold." << std::endl;
    }
    return goodMatches;
}

/**
 * @param image to be compared to the image library
 * @return a vector of the best matches
 * Preconditions: The image must be a valid jpg image and the image library must be populated.
 * Postconditions: The vector will contain the best matches from the image library determined using SIFT detectors.
 */
std::vector<cv::Mat> ImageMatcher::siftMatch(const cv::Mat &image) {
    std::vector<cv::Mat> bestMatches;
    if(image.empty() || imageLibrary.empty()){
        return bestMatches;
    }

    //timer for metrics
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // Compute keypoints and descriptors for the input image
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    siftDetector->detectAndCompute(image, noArray(), keypoints, descriptors);

    for (int i = 0; i < imageLibrary.size(); i++){ // for each image in the library
        // Display progress
        std::cout << "Processing image " << i+1 << " of " << imageLibrary.size() << std::endl;
        // Compute keypoints and descriptors for the library image
        Mat libraryDescriptors;
        siftDetector->detectAndCompute(imageLibrary[i], noArray(), siftKeypoints[i], libraryDescriptors, true);
        std::vector<DMatch> goodMatches = scoreMatches(descriptors, libraryDescriptors, bestMatches, i);
        Mat output;
        drawMatches(image, keypoints, imageLibrary[i], siftKeypoints[i], goodMatches, output);
        imshow("Matches", output);
        waitKey(0);
    }

    //print timer
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time elapsed: " << float(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count())/1000 << "seconds" << std::endl;

    return bestMatches;
}

/**
 * @param image to be compared to the image library
 * @param m is the number of sectors to use for sGLOH2
 * @return a vector of the best matches
 * Preconditions: The image must be a valid jpg image and the image library must be populated.
 * Postconditions: The vector will contain the best matches from the image library determined using sGLOH2 detectors.
 */
std::vector<cv::Mat> ImageMatcher::sGLOHMatch(const cv::Mat &image, int m) {
    // Why does everything pass?

    std::vector<cv::Mat> bestMatches;
    if (image.empty() || imageLibrary.empty()) {
        return bestMatches;
    }

    //timer for metrics
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // Compute keypoints and descriptors for the input image
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    sGLOH2 sgloh2(m);
    sgloh2.compute(image, keypoints, descriptors); //This calculates keypoints again so calculating in the constructor doesn't help

    for (int i = 0; i < imageLibrary.size(); i++) { // for each image in the library
        // Display progress
        std::cout << "Processing image " << i + 1 << " of " << imageLibrary.size() << std::endl;
        // Compute keypoints and descriptors for the library image
        Mat libraryDescriptors;
        sgloh2.compute(imageLibrary[i], siftKeypoints[i],
                       libraryDescriptors);  // Currently using siftKeypoints for sGLOH2 descriptors
        std::vector<DMatch> goodMatches = scoreMatches(descriptors, libraryDescriptors, bestMatches, i);
        Mat output;
        drawMatches(image, keypoints, imageLibrary[i], siftKeypoints[i], goodMatches, output);
        imshow("Matches", output);
        waitKey(0);
    }

    //print timer
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time elapsed: "
              << float(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()) / 1000 << " ms"
              << std::endl;

    return bestMatches;
}
