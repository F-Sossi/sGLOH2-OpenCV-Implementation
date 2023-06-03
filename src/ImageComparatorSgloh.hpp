/**
 * @file ImageComparatorSgloh.hpp
 * @author Frank Sossi, Justin Boyer
 * @date 6/1/23
 * @brief Definition of the ImageComparatorSgloh class.
 *
 * This class provides the definition for the ImageComparatorSgloh class, including
 * constructors, computation and utility methods. The class is designed
 * for comparing images using the sGLOH2 descriptor.
 *
 * Implemented functions:
 * ImageComparatorSgloh::ImageComparatorSgloh(std::string  inputImagePath, std::string  folderPath)
 * void ImageComparatorSgloh::runComparison(bool suppressInput = false)
 * cv::Rect ImageComparatorSgloh::selectROI(const cv::Mat& image)
 *
 * This work is based on the following paper:
 * Bellavia, Fabio, and Carlo Colombo. "Rethinking the sGLOH descriptor." IEEE Transactions on Pattern
 * Analysis and Machine Intelligence 40.4 (2017): 931-944.
 *
 * Bellavia, Fabio, Domenico Tegolo, and Emanuele Trucco. "Improving SIFT-based descriptors stability
 * to rotations." 2010 20th International Conference on Pattern Recognition. IEEE, 2010.
 *
 */

#ifndef SGLOH_OPENCV_IMAGECOMPARATORSGLOH_HPP
#define SGLOH_OPENCV_IMAGECOMPARATORSGLOH_HPP

#include "sGLOH2.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <filesystem>
#include <utility>
#include <vector>
#include <queue>
#include <limits>
#include <mutex>
#include <chrono>

namespace fs = std::filesystem;

// Number of images to match
constexpr int NUM_MATCHES_SGLOH = 5;

// Max distance between keypoints to be considered a match
constexpr double MAX_DISTANCE_SGLOH = 0.4;

class ImageComparatorSgloh {
public:
    /**
     * @brief Constructor for the ImageComparatorSgloh class.
     * @param inputImagePath The path to the input image.
     * @param folderPath The path to the folder containing the images to compare to.
     */
    ImageComparatorSgloh(std::string  inputImagePath, std::string  folderPath)
            : inputImagePath_(std::move(inputImagePath)), folderPath_(std::move(folderPath)) {}

    /**
     * @brief Runs the sGLOH2 descriptor on the input image and compares it to the images in the folder.
     * @param suppressInput If true, the user will not be prompted to select a region of interest.
     * If false, the user will be prompted to select a region of interest.
     */
    void runComparison(bool suppressInput = false) {
        // Load the input image
        cv::Mat inputImage = cv::imread(inputImagePath_, cv::IMREAD_GRAYSCALE);

        // Check if the image was loaded successfully
        if (inputImage.empty()) {
            std::cerr << "Error: Failed to load image: " << inputImagePath_ << std::endl;
            return;
        }

        if(!suppressInput) {
            // Select a ROI from the input image
            cv::Rect roi = selectROI(inputImage);

            // Only keep the part of the image within the ROI
            inputImage = inputImage(roi);
        }

        auto start = std::chrono::high_resolution_clock::now();
        int imageCount = 1;

        // Initialize sGLOH2 descriptor
        sGLOH2 sgloh2;

        // Compute sGLOH2 descriptors for the input image
        std::vector<cv::KeyPoint> inputKeyPoints;
        cv::Mat inputDescriptors;
        sgloh2.compute(inputImage, inputKeyPoints, inputDescriptors);

        // Get all image paths in the folder
        std::vector<std::string> imagePaths;
        for (const auto& entry : fs::directory_iterator(folderPath_)) {
            imagePaths.push_back(entry.path().string());
        }

        // Create a mutex for synchronization
        std::mutex mtx;

        // Process images in parallel
        cv::parallel_for_(cv::Range(0, imagePaths.size()), [&](const cv::Range& range) {
            size_t start = size_t(range.start); // use size_t to avoid warning
            size_t end = size_t(range.end);
            for (size_t i = start; i < end; i++) {
                // Load image
                cv::Mat image = cv::imread(imagePaths[i], cv::IMREAD_GRAYSCALE);

                // Compute descriptors
                std::vector<cv::KeyPoint> keyPoints;
                cv::Mat descriptors;
                sgloh2.compute(image, keyPoints, descriptors);

                // Compute matches
                std::vector<cv::DMatch> matches_sgloh2;
                cv::parallel_for_(cv::Range(0, inputDescriptors.rows), [&](const cv::Range& range) {
                    std::vector<cv::DMatch> matches_sgloh2_local;
                    for (int i = range.start; i < range.end; i++) {
                        double min_distance_sgloh2 = std::numeric_limits<double>::max();
                        int best_match = -1;
                        for (int j = 0; j < descriptors.rows; ++j) {
                            double distance = sgloh2.distance(inputDescriptors.row(i), descriptors.row(j));
                            if (distance < min_distance_sgloh2) {
                                min_distance_sgloh2 = distance;
                                best_match = j;
                            }
                        }
                        if (best_match != -1) {
                            matches_sgloh2_local.emplace_back(i, best_match, min_distance_sgloh2);
                        }
                    }

                    // Synchronize access to the matches_sgloh2 vector
                    std::lock_guard<std::mutex> lock(mtx);
                    matches_sgloh2.insert(matches_sgloh2.end(), matches_sgloh2_local.begin(), matches_sgloh2_local.end());
                });


                // Filter out good matches based on their distance
                std::vector<cv::DMatch> good_matches_sgloh2;
                float max_distance = MAX_DISTANCE_SGLOH;
                for (const auto& match : matches_sgloh2) {
                    if (match.distance <= max_distance) {
                        good_matches_sgloh2.push_back(match);
                    }
                }


                // Synchronize access to the priority queue
                std::lock_guard<std::mutex> lock(mtx);
                ImageMatches imageMatches;
                imageMatches.path = imagePaths[i];
                imageMatches.count = good_matches_sgloh2.size();
                mostMatches.push(imageMatches);

                // Store keypoints and matches for each image
                keypointsMap[imagePaths[i]] = keyPoints;
                matchesMap[imagePaths[i]] = matches_sgloh2;

                // Print progress
                std::cout << "Processed image: " << imageCount << std::endl;
                imageCount++;
            }
        });


        //print exit for loop
        std::cout << "Finished processing images" << std::endl;
        // end timer
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        std::cout << "sGLOH2 descriptor took " << elapsed.count() << " seconds" << std::endl;

        // Draw matches between input image and top images
        std::vector<cv::Mat> topImages;
        std::vector<std::vector<cv::DMatch>> topMatches;
        std::vector<std::vector<cv::KeyPoint>> topKeypoints;

        for (int i = 0; i < NUM_MATCHES_SGLOH && !mostMatches.empty(); ++i) {
            const auto& top = mostMatches.top();
            topImages.push_back(cv::imread(top.path, cv::IMREAD_GRAYSCALE));
            topMatches.push_back(matchesMap[top.path]);
            topKeypoints.push_back(keypointsMap[top.path]);
            mostMatches.pop();
        }

        for (size_t i = 0; i < topImages.size(); ++i) {
            cv::Mat imgMatches;
            cv::drawMatches(inputImage, inputKeyPoints, topImages[i], topKeypoints[i], topMatches[i], imgMatches);

            // Create a unique window name for each match
            std::string windowName = "Match " + std::to_string(i + 1);
            cv::imshow(windowName, imgMatches);
        }

        cv::waitKey(0);

    }

private:

    /**
     * @brief Selects a Region of Interest (ROI) from the input image.
     * @param image The input image.
     * @return The selected ROI.
     */
    cv::Rect selectROI(const cv::Mat& image) {
        // Display the image and wait for a rectangle selection
        cv::Rect roi = cv::selectROI("Select ROI", image);

        // Destroy the "Select ROI" window
        cv::destroyWindow("Select ROI");

        return roi;
    };

    /**
     * @struct ImageMatches
     * @brief This struct is used to store the image path and the count of matches.
     */
    struct ImageMatches {
        std::string path;
        size_t count;

        // This makes the priority queue a max-heap instead of a min-heap
        bool operator<(const ImageMatches& other) const {
            return count < other.count;
        }
    };

    std::string inputImagePath_;
    std::string folderPath_;

    // Declare a priority queue to hold the images with the most matches
    std::priority_queue<ImageMatches> mostMatches;

    std::map<std::string, std::vector<cv::KeyPoint>> keypointsMap;
    std::map<std::string, std::vector<cv::DMatch>> matchesMap;

};

#endif //SGLOH_OPENCV_IMAGECOMPARATORSGLOH_HPP
/**
 * This work is based on the following paper:
 * Bellavia, Fabio, and Carlo Colombo. "Rethinking the sGLOH descriptor." IEEE Transactions on Pattern
 * Analysis and Machine Intelligence 40.4 (2017): 931-944.
 *
 * Bellavia, Fabio, Domenico Tegolo, and Emanuele Trucco. "Improving SIFT-based descriptors stability
 * to rotations." 2010 20th International Conference on Pattern Recognition. IEEE, 2010.
 *
 */
