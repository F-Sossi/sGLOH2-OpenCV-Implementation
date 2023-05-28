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

namespace fs = std::filesystem;

class ImageComparatorSgloh {
public:
    ImageComparatorSgloh(std::string  inputImagePath, std::string  folderPath)
            : inputImagePath_(std::move(inputImagePath)), folderPath_(std::move(folderPath)) {}

    void runComparison() {
        // Load the input image
        cv::Mat inputImage = cv::imread(inputImagePath_, cv::IMREAD_GRAYSCALE);

        // Select a ROI from the input image
        cv::Rect roi = selectROI(inputImage);

        // Only keep the part of the image within the ROI
        inputImage = inputImage(roi);

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
            for (int i = range.start; i < range.end; i++) {
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
                double max_distance = 0.4; // We can adjust this value to find a good threshold for our dataset
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
                std::cout << "Processed " << range.end - range.start << " images" << std::endl;
            }

        });


        //print exit for loop
        std::cout << "Finished processing images" << std::endl;

        // Draw matches between input image and top three images
        std::vector<cv::Mat> topImages;
        std::vector<std::vector<cv::DMatch>> topMatches;
        std::vector<std::vector<cv::KeyPoint>> topKeypoints;
        for (int i = 0; i < 5 && !mostMatches.empty(); ++i) {
            const auto& top = mostMatches.top();
            topImages.push_back(cv::imread(top.path, cv::IMREAD_GRAYSCALE));
            topMatches.push_back(matchesMap[top.path]);
            topKeypoints.push_back(keypointsMap[top.path]);
            mostMatches.pop();
        }

        for (size_t i = 0; i < topImages.size(); ++i) {
//            cv::Mat imgMatches;
//            cv::drawMatches(inputImage, inputKeyPoints, topImages[i], topKeypoints[i], topMatches[i], imgMatches);

            // Create a unique window name for each match
            std::string windowName = "Match " + std::to_string(i+1);
            cv::imshow(windowName, topImages[i]);
        }

        cv::waitKey(0);

    }

private:
    cv::Rect selectROI(const cv::Mat& image) {
        // Display the image and wait for a rectangle selection
        cv::Rect roi = cv::selectROI("Select ROI", image);

        // Destroy the "Select ROI" window
        cv::destroyWindow("Select ROI");

        return roi;
    };

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
