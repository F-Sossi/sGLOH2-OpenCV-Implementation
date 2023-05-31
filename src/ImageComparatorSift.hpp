#ifndef SIFT_OPENCV_IMAGECOMPARATORSIFT_HPP
#define SIFT_OPENCV_IMAGECOMPARATORSIFT_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <filesystem>
#include <utility>
#include <vector>
#include <queue>
#include <limits>
#include <mutex>

namespace fs = std::filesystem;

class ImageComparatorSift {
public:
    ImageComparatorSift(std::string  inputImagePath, std::string  folderPath)
            : inputImagePath_(std::move(inputImagePath)), folderPath_(std::move(folderPath)) {}

    void runComparison(bool suppressInput = false) {
        // Load the input image
        cv::Mat inputImage = cv::imread(inputImagePath_, cv::IMREAD_GRAYSCALE);

        // Check if the image was loaded successfully
        if (inputImage.empty()) {
            std::cerr << "Error: Failed to load image: " << inputImagePath_ << std::endl;
            return;
        }

        // Initialize SIFT descriptor
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

        // Compute SIFT descriptors for the input image
        std::vector<cv::KeyPoint> inputKeyPoints;
        cv::Mat inputDescriptors;
        sift->detectAndCompute(inputImage, cv::noArray(), inputKeyPoints, inputDescriptors);

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
                sift->detectAndCompute(image, cv::noArray(), keyPoints, descriptors);

                // Compute matches
                cv::BFMatcher matcher(cv::NORM_L2);
                std::vector<cv::DMatch> matches;
                matcher.match(inputDescriptors, descriptors, matches);

                // Filter out good matches based on their distance
                std::vector<cv::DMatch> good_matches;
                double max_distance = 0.4; // We can adjust this value to find a good threshold for our dataset
                for (const auto& match : matches) {
                    if (match.distance <= max_distance) {
                        good_matches.push_back(match);
                    }
                }

                // Synchronize access to the priority queue
                std::lock_guard<std::mutex> lock(mtx);
                ImageMatches imageMatches;
                imageMatches.path = imagePaths[i];
                imageMatches.count = good_matches.size();
                mostMatches.push(imageMatches);

                // Store keypoints and matches for each image
                keypointsMap[imagePaths[i]] = keyPoints;
                matchesMap[imagePaths[i]] = good_matches;

                // Print progress
                std::cout << "Processed image: " << imageCount << std::endl;
                imageCount++;
            }

        });

        // Print exit for loop
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
        if(!suppressInput) {
            for (size_t i = 0; i < topImages.size(); ++i) {
                cv::Mat imgMatches;
                cv::drawMatches(inputImage, inputKeyPoints, topImages[i], topKeypoints[i], topMatches[i], imgMatches);

                // Create a unique window name for each match
                std::string windowName = "Match " + std::to_string(i + 1);
                cv::imshow(windowName, imgMatches);
            }
            cv::waitKey(0);
        }

    }

private:
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

#endif //SIFT_OPENCV_IMAGECOMPARATORSIFT_HPP

