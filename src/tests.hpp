//
// Created by user on 5/13/23.
//

#ifndef SGLOH_OPENCV_TESTS_HPP
#define SGLOH_OPENCV_TESTS_HPP

#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include "sGLOH2.hpp"

//constexpr int M = 8;
constexpr int sGLOH_THRESHOLD = 20;

void compareImages(const std::string& filename1, const std::string& filename2) {
    // Load the images in grayscale
    cv::Mat image1 = cv::imread(filename1, cv::IMREAD_GRAYSCALE);
    cv::Mat image2 = cv::imread(filename2, cv::IMREAD_GRAYSCALE);

    // Initialize SIFT detector and compute keypoints and descriptors for both images
    // start timer
    auto start = std::chrono::high_resolution_clock::now();
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints_sift1, keypoints_sift2;
    cv::Mat descriptors_sift1, descriptors_sift2;
    sift->detectAndCompute(image1, cv::noArray(), keypoints_sift1, descriptors_sift1);
    sift->detectAndCompute(image2, cv::noArray(), keypoints_sift2, descriptors_sift2);
    // end timer
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "SIFT time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // Match the SIFT descriptors using Brute-Force matcher
    cv::BFMatcher matcher_sift;
    std::vector<cv::DMatch> matches_sift;
    matcher_sift.match(descriptors_sift1, descriptors_sift2, matches_sift);

    // Filter out good matches based on their distance
    std::vector<cv::DMatch> good_matches_sift;
    double min_dist_sift = 100;
    for( int i = 0; i < descriptors_sift1.rows; i++ ) {
        if( matches_sift[i].distance <= std::max(2*min_dist_sift, 0.02) ) {
            good_matches_sift.push_back( matches_sift[i]);
        }
    }

    // Initialize sGLOH2 descriptor and compute keypoints and descriptors for both images
    // start timer
    start = std::chrono::high_resolution_clock::now();
    sGLOH2 sgloh2;
    std::vector<cv::KeyPoint> keypoints_sgloh2_1, keypoints_sgloh2_2;
    cv::Mat descriptors_sgloh2_1, descriptors_sgloh2_2;
    sgloh2.compute(image1, keypoints_sgloh2_1, descriptors_sgloh2_1);
    sgloh2.compute(image2, keypoints_sgloh2_2, descriptors_sgloh2_2);
    // end timer
    end = std::chrono::high_resolution_clock::now();
    std::cout << "sGLOH2 time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // Initialize vectors to hold matches
    std::vector<std::vector<cv::DMatch>> matches_sgloh2_threads;

    // Use OpenCV's parallel_for_ to run loops in parallel
    cv::parallel_for_(cv::Range(0, descriptors_sgloh2_1.rows), [&](const cv::Range& range) {
        std::vector<cv::DMatch> matches_sgloh2_local;
        for (int i = range.start; i < range.end; i++) {
            double min_distance_sgloh2 = std::numeric_limits<double>::max();
            int best_match = -1;
            for (int j = 0; j < descriptors_sgloh2_2.rows; ++j) {
                double distance = sgloh2.distance(descriptors_sgloh2_1.row(i), descriptors_sgloh2_2.row(j));
                if (distance < min_distance_sgloh2) {
                    min_distance_sgloh2 = distance;
                    best_match = j;
                }
            }
            if (best_match != -1) {
                matches_sgloh2_local.push_back(cv::DMatch(i, best_match, min_distance_sgloh2));
            }
        }
        matches_sgloh2_threads.push_back(matches_sgloh2_local);
    });

    // Merge all local matches vectors
    std::vector<cv::DMatch> matches_sgloh2;
    for (const auto& matches : matches_sgloh2_threads) {
        matches_sgloh2.insert(matches_sgloh2.end(), matches.begin(), matches.end());
    }

//    // Filter out good matches based on their distance
//    std::vector<cv::DMatch> good_matches_sgloh2;
//    auto max_distance = 0.4;
//    for (const auto& match : matches_sgloh2) {
//        if (match.distance <= max_distance) {
//            good_matches_sgloh2.push_back(match);
//        }
//    }

// Filter out good matches based on their distance
    std::vector<cv::DMatch> good_matches_sgloh2;
    double min_dist_sgloh2 = 20; // You may need to adjust this value based on your specific application
    for (const auto& match : matches_sgloh2) {
        if (match.distance <= std::max(2*min_dist_sgloh2, 0.02)) {
            good_matches_sgloh2.push_back(match);
        }
    }



    // Print the number of good matches for both SIFT and sGLOH2
    std::cout << "Number of good matches with sGLOH2: " << good_matches_sgloh2.size() << std::endl;
    std::cout << "Number of good matches with SIFT: " << good_matches_sift.size() << std::endl;

    // Draw the good matches for both SIFT
    cv::Mat img_matches_sift, img_matches_sgloh2;
    cv::drawMatches(image1, keypoints_sift1, image2, keypoints_sift2, good_matches_sift, img_matches_sift);
    cv::drawMatches(image1, keypoints_sgloh2_1, image2, keypoints_sgloh2_2, good_matches_sgloh2, img_matches_sgloh2);

    cv::imshow("Good Matches SIFT", img_matches_sift);
    cv::imshow("Good Matches sGLOH2", img_matches_sgloh2);
    cv::waitKey(0);
}


// tests using a rotation of the image
void processImage(const std::string& filename) {
    // Load the image in grayscale
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    cv::Mat image_flipped;

    // Rotate the image by 180 degrees
    cv::rotate(image, image_flipped, cv::ROTATE_180);

    // Initialize SIFT detector and compute keypoints and descriptors for both images
    // start timer
    auto start = std::chrono::high_resolution_clock::now();
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints_sift, keypoints_sift_flipped;
    cv::Mat descriptors_sift, descriptors_sift_flipped;
    sift->detectAndCompute(image, cv::noArray(), keypoints_sift, descriptors_sift);
    sift->detectAndCompute(image_flipped, cv::noArray(), keypoints_sift_flipped, descriptors_sift_flipped);
    // end timer
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "SIFT time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // Match the SIFT descriptors using Brute-Force matcher
    cv::BFMatcher matcher_sift;
    std::vector<cv::DMatch> matches_sift;
    matcher_sift.match(descriptors_sift, descriptors_sift_flipped, matches_sift);

    // Filter out good matches based on their  distance
    std::vector<cv::DMatch> good_matches_sift;
    double min_dist_sift = 100;
    for( int i = 0; i < descriptors_sift.rows; i++ ) {
        if( matches_sift[i].distance <= std::max(2*min_dist_sift, 0.02) ) {
            good_matches_sift.push_back( matches_sift[i]);
        }
    }

    // Initialize sGLOH2 descriptor and compute keypoints and descriptors for both images
    // start timer
    start = std::chrono::high_resolution_clock::now();
    sGLOH2 sgloh2;
    std::vector<cv::KeyPoint> keypoints_sgloh2, keypoints_sgloh2_flipped;
    cv::Mat descriptors_sgloh2, descriptors_sgloh2_flipped;
    sgloh2.compute(image, keypoints_sgloh2, descriptors_sgloh2);
    sgloh2.compute(image_flipped, keypoints_sgloh2_flipped, descriptors_sgloh2_flipped);
    // end timer
    end = std::chrono::high_resolution_clock::now();
    std::cout << "sGLOH2 time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // Initialize vectors to hold matches
    std::vector<std::vector<cv::DMatch>> matches_sgloh2_threads;

    // Use OpenCV's parallel_for_ to run loops in parallel
    cv::parallel_for_(cv::Range(0, descriptors_sgloh2.rows), [&](const cv::Range& range) {
        std::vector<cv::DMatch> matches_sgloh2_local;
        for (int i = range.start; i < range.end; i++) {
            double min_distance_sgloh2 = std::numeric_limits<double>::max();
            int best_match = -1;
            for (int j = 0; j < descriptors_sgloh2_flipped.rows; ++j) {
                double distance = sgloh2.distance(descriptors_sgloh2.row(i), descriptors_sgloh2_flipped.row(j));
                if (distance < min_distance_sgloh2) {
                    min_distance_sgloh2 = distance;
                    best_match = j;
                }
            }
            if (best_match != -1) {
                matches_sgloh2_local.push_back(cv::DMatch(i, best_match, min_distance_sgloh2));
            }
        }
        matches_sgloh2_threads.push_back(matches_sgloh2_local);
    });

    // Merge all local matches vectors
    std::vector<cv::DMatch> matches_sgloh2;
    for (const auto& matches : matches_sgloh2_threads) {
        matches_sgloh2.insert(matches_sgloh2.end(), matches.begin(), matches.end());
    }

    // Filter out good matches based on their distance
    std::vector<cv::DMatch> good_matches_sgloh2;
    auto max_distance = 0.4;
    for (const auto& match : matches_sgloh2) {
        if (match.distance <= max_distance) {
            good_matches_sgloh2.push_back(match);
        }
    }


    // Print the number of good matches for both SIFT and sGLOH2
    std::cout << "Number of good matches with sGLOH2: " << good_matches_sgloh2.size() << std::endl;
    std::cout << "Number of good matches with SIFT: " << good_matches_sift.size() << std::endl;

    // Draw the good matches for both SIFT
    cv::Mat img_matches_sift, img_matches_sgloh2;
    cv::drawMatches(image, keypoints_sift, image_flipped, keypoints_sift_flipped, good_matches_sift, img_matches_sift);
    cv::drawMatches(image, keypoints_sgloh2, image_flipped, keypoints_sgloh2_flipped, good_matches_sgloh2, img_matches_sgloh2);

    cv::imshow("Good Matches SIFT", img_matches_sift);
    cv::imshow("Good Matches sGLOH2", img_matches_sgloh2);
    cv::waitKey(0);
}

// simple test using identical images
void processImage2(const std::string& imageFileName) {
    // Load the image in grayscale
    cv::Mat image = cv::imread(imageFileName, cv::IMREAD_GRAYSCALE);
    cv::Mat image_copy = image.clone();

    // Initialize SIFT detector and compute keypoints and descriptors for both images
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints_sift, keypoints_sift_copy;
    cv::Mat descriptors_sift, descriptors_sift_copy;
    sift->detectAndCompute(image, cv::noArray(), keypoints_sift, descriptors_sift);
    sift->detectAndCompute(image_copy, cv::noArray(), keypoints_sift_copy, descriptors_sift_copy);

    // Match the SIFT descriptors using Brute-Force matcher
    cv::BFMatcher matcher_sift;
    std::vector<cv::DMatch> matches_sift;
    matcher_sift.match(descriptors_sift, descriptors_sift_copy, matches_sift);

    // Filter out good matches based on their distance
    std::vector<cv::DMatch> good_matches_sift;
    double min_dist_sift = sGLOH_THRESHOLD;
    for( int i = 0; i < descriptors_sift.rows; i++ ) {
        if( matches_sift[i].distance <= std::max(2*min_dist_sift, 0.02) ) {
            good_matches_sift.push_back(matches_sift[i]);
        }
    }

    // Initialize sGLOH2 descriptor and compute keypoints and descriptors for both images
    sGLOH2 sgloh2;
    std::vector<cv::KeyPoint> keypoints_sgloh2, keypoints_sgloh2_copy;
    cv::Mat descriptors_sgloh2, descriptors_sgloh2_copy;
    sgloh2.compute(image, keypoints_sgloh2, descriptors_sgloh2);
    sgloh2.compute(image_copy, keypoints_sgloh2_copy, descriptors_sgloh2_copy);

    // Match the sGLOH2 descriptors using Brute-Force matcher
    cv::BFMatcher matcher_sgloh2(cv::NORM_L2);
    std::vector<cv::DMatch> matches_sgloh2;
    matcher_sgloh2.match(descriptors_sgloh2, descriptors_sgloh2_copy, matches_sgloh2);

    // Filter out good matches based on their distance
    std::vector<cv::DMatch> good_matches_sgloh2;
    double min_dist_sgloh2 = sGLOH_THRESHOLD;
    for( int i = 0; i < descriptors_sgloh2.rows; i++ ) {
        if( matches_sgloh2[i].distance <= std::max(2*min_dist_sgloh2, 0.02) ) {
            good_matches_sgloh2.push_back(matches_sgloh2[i]);
        }
    }

    // Print the number of good matches for both SIFT and sGLOH2
    std::cout << "Number of good matches with sGLOH2: " << good_matches_sgloh2.size() << std::endl;
    std::cout << "Number of good matches with SIFT: " << good_matches_sift.size() << std::endl;

    // Draw the good matches for both SIFT
    cv::Mat img_matches_sift, img_matches_sgloh2;
    cv::drawMatches(image, keypoints_sift, image_copy, keypoints_sift_copy, good_matches_sift, img_matches_sift);
    cv::drawMatches(image, keypoints_sgloh2,    // Continue from the previous part
                    image_copy, keypoints_sgloh2_copy, good_matches_sgloh2, img_matches_sgloh2);

    cv::imshow("Good Matches SIFT", img_matches_sift);
    cv::imshow("Good Matches sGLOH2", img_matches_sgloh2);
    cv::waitKey(0);
}

// test with shifted image (horizontal shift)
void processImage3(const std::string& imageFileName) {
    // Load the image in grayscale
    cv::Mat image = cv::imread(imageFileName, cv::IMREAD_GRAYSCALE);
    cv::Mat image_shifted;

    // Create the transformation matrix for shifting
    cv::Mat trans_mat = (cv::Mat_<double>(2,3) << 1, 0, 100, 0, 1, 0); // Shifts the image 100 pixels to the right
    cv::warpAffine(image, image_shifted, trans_mat, image.size());

    // Initialize SIFT detector and compute keypoints and descriptors for both images
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints_sift, keypoints_sift_copy;
    cv::Mat descriptors_sift, descriptors_sift_copy;
    sift->detectAndCompute(image, cv::noArray(), keypoints_sift, descriptors_sift);
    sift->detectAndCompute(image_shifted, cv::noArray(), keypoints_sift_copy, descriptors_sift_copy);

    // Match the SIFT descriptors using Brute-Force matcher
    cv::BFMatcher matcher_sift;
    std::vector<cv::DMatch> matches_sift;
    matcher_sift.match(descriptors_sift, descriptors_sift_copy, matches_sift);

    // Filter out good matches based on their distance
    std::vector<cv::DMatch> good_matches_sift;
    double min_dist_sift = 100;
    for( int i = 0; i < descriptors_sift.rows; i++ ) {
        if( matches_sift[i].distance <= std::max(2*min_dist_sift, 0.02) ) {
            good_matches_sift.push_back(matches_sift[i]);
        }
    }

    // Initialize sGLOH2 descriptor and compute keypoints and descriptors for both images
    sGLOH2 sgloh2;
    std::vector<cv::KeyPoint> keypoints_sgloh2, keypoints_sgloh2_copy;
    cv::Mat descriptors_sgloh2, descriptors_sgloh2_copy;
    sgloh2.compute(image, keypoints_sgloh2, descriptors_sgloh2);
    sgloh2.compute(image_shifted, keypoints_sgloh2_copy, descriptors_sgloh2_copy);

    // Match the sGLOH2 descriptors using Brute-Force matcher
    cv::BFMatcher matcher_sgloh2(cv::NORM_L2);
    std::vector<cv::DMatch> matches_sgloh2;
    matcher_sgloh2.match(descriptors_sgloh2, descriptors_sgloh2_copy, matches_sgloh2);

    // Filter out good matches based on their distance
    std::vector<cv::DMatch> good_matches_sgloh2;
    double min_dist_sgloh2 = sGLOH_THRESHOLD;
    for( int i = 0; i < descriptors_sgloh2.rows; i++ ) {
        if( matches_sgloh2[i].distance <= std::max(2*min_dist_sgloh2, 0.02) ) {
            good_matches_sgloh2.push_back(matches_sgloh2[i]);
        }
    }

    // Print the number of good matches for both SIFT and sGLOH2
    std::cout << "Number of good matches with sGLOH2: " << good_matches_sgloh2.size() << std::endl;
    std::cout << "Number of good matches with SIFT: " << good_matches_sift.size() << std::endl;

    // Draw the good matches for both SIFT
    cv::Mat img_matches_sift, img_matches_sgloh2;
    cv::drawMatches(image, keypoints_sift, image_shifted, keypoints_sift_copy, good_matches_sift, img_matches_sift);
    cv::drawMatches(image, keypoints_sgloh2,    // Continue from the previous part
                    image_shifted, keypoints_sgloh2_copy, good_matches_sgloh2, img_matches_sgloh2);

    cv::imshow("Good Matches SIFT", img_matches_sift);
    cv::imshow("Good Matches sGLOH2", img_matches_sgloh2);
    cv::waitKey(0);
}

void processImage4(const std::string& filename) {
    // Load the image in grayscale
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    cv::Mat image_flipped;

    // Rotate the image by 180 degrees
    cv::rotate(image, image_flipped, cv::ROTATE_180);

    // Initialize SIFT detector and compute keypoints and descriptors for both images
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints_sift, keypoints_sift_flipped;
    cv::Mat descriptors_sift, descriptors_sift_flipped;
    sift->detectAndCompute(image, cv::noArray(), keypoints_sift, descriptors_sift);
    sift->detectAndCompute(image_flipped, cv::noArray(), keypoints_sift_flipped, descriptors_sift_flipped);

    // Match the SIFT descriptors using Brute-Force matcher
    cv::BFMatcher matcher_sift;
    std::vector<cv::DMatch> matches_sift;
    matcher_sift.match(descriptors_sift, descriptors_sift_flipped, matches_sift);

    // Filter out good matches based on their distance
    std::vector<cv::DMatch> good_matches_sift;
    double min_dist_sift = 100;
    for( int i = 0; i < descriptors_sift.rows; i++ ) {
        if( matches_sift[i].distance <= std::max(2*min_dist_sift, 0.02) ) {
            good_matches_sift.push_back( matches_sift[i]);
        }
    }

    // Initialize sGLOH2 descriptor and compute keypoints and descriptors for both images
    sGLOH2 sgloh2;
    std::vector<cv::KeyPoint> keypoints_sgloh2, keypoints_sgloh2_flipped;
    cv::Mat descriptors_sgloh2, descriptors_sgloh2_flipped;
    sgloh2.compute(image, keypoints_sgloh2, descriptors_sgloh2);
    sgloh2.compute(image_flipped, keypoints_sgloh2_flipped, descriptors_sgloh2_flipped);

    // Match the sGLOH2 descriptors using FLANN matcher
    cv::FlannBasedMatcher matcher_sgloh2;
    std::vector<cv::DMatch> matches_sgloh2;
    matcher_sgloh2.match(descriptors_sgloh2, descriptors_sgloh2_flipped, matches_sgloh2);

    // Filter out good matches based on their distance
    std::vector<cv::DMatch> good_matches_sgloh2;
    double min_dist_sgloh2 = sGLOH_THRESHOLD;
    for( int i = 0; i < descriptors_sgloh2.rows; i++ ) {
        if( matches_sgloh2[i].distance <= std::max(2*min_dist_sgloh2, 0.02) ) {
            good_matches_sgloh2.push_back( matches_sgloh2[i]);
        }
    }

    // Print the number of good matches for both SIFT and sGLOH2
    std::cout << "Number of good matches with sGLOH2: " << good_matches_sgloh2.size() << std::endl;
    std::cout << "Number of good matches with SIFT: " << good_matches_sift.size() << std::endl;

    // Draw the good matches for both SIFT
    cv::Mat img_matches_sift, img_matches_sgloh2;
    cv::drawMatches(image, keypoints_sift, image_flipped, keypoints_sift_flipped, good_matches_sift, img_matches_sift);
    cv::drawMatches(image, keypoints_sgloh2, image_flipped, keypoints_sgloh2_flipped, good_matches_sgloh2, img_matches_sgloh2);

    cv::imshow("Good Matches SIFT", img_matches_sift);
    cv::imshow("Good Matches sGLOH2", img_matches_sgloh2);
    cv::waitKey(0);
}

void processImage5(const std::string& imageFileName) {
    // Load the image in grayscale
    cv::Mat image = cv::imread(imageFileName, cv::IMREAD_GRAYSCALE);
    cv::Mat image_scaled;

    // Scale the image down by 50%
    cv::resize(image, image_scaled, cv::Size(), 0.5, 0.5);

    // Initialize SIFT detector and compute keypoints and descriptors for both images
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints_sift, keypoints_sift_copy;
    cv::Mat descriptors_sift, descriptors_sift_copy;
    sift->detectAndCompute(image, cv::noArray(), keypoints_sift, descriptors_sift);
    sift->detectAndCompute(image_scaled, cv::noArray(), keypoints_sift_copy, descriptors_sift_copy);

    // Match the SIFT descriptors using Brute-Force matcher
    cv::BFMatcher matcher_sift;
    std::vector<cv::DMatch> matches_sift;
    matcher_sift.match(descriptors_sift, descriptors_sift_copy, matches_sift);

    // Filter out good matches based on their distance
    std::vector<cv::DMatch> good_matches_sift;
    double min_dist_sift = 100;
    for( int i = 0; i < descriptors_sift.rows; i++ ) {
        if( matches_sift[i].distance <= std::max(2*min_dist_sift, 0.02) ) {
            good_matches_sift.push_back(matches_sift[i]);
        }
    }

    // Initialize sGLOH2 descriptor and compute keypoints and descriptors for both images
    sGLOH2 sgloh2;
    std::vector<cv::KeyPoint> keypoints_sgloh2, keypoints_sgloh2_copy;
    cv::Mat descriptors_sgloh2, descriptors_sgloh2_copy;
    sgloh2.compute(image, keypoints_sgloh2, descriptors_sgloh2);
    sgloh2.compute(image_scaled, keypoints_sgloh2_copy, descriptors_sgloh2_copy);

    // Match the sGLOH2 descriptors using Brute-Force matcher
    cv::BFMatcher matcher_sgloh2(cv::NORM_L2);
    std::vector<cv::DMatch> matches_sgloh2;
    matcher_sgloh2.match(descriptors_sgloh2, descriptors_sgloh2_copy, matches_sgloh2);

    // Filter out good matches based on their distance
    std::vector<cv::DMatch> good_matches_sgloh2;
    double min_dist_sgloh2 = sGLOH_THRESHOLD;
    for( int i = 0; i < descriptors_sgloh2.rows; i++ ) {
        if( matches_sgloh2[i].distance <= std::max(2*min_dist_sgloh2, 0.02) ) {
            good_matches_sgloh2.push_back(matches_sgloh2[i]);
        }
    }

    // Print the number of good matches for both SIFT and sGLOH2
    std::cout << "Number of good matches with sGLOH2: " << good_matches_sgloh2.size() << std::endl;
    std::cout << "Number of good matches with SIFT: " << good_matches_sift.size() << std::endl;

    // Draw the good matches for both SIFT and sGLOH2
    cv::Mat img_matches_sift, img_matches_sgloh2;
    cv::drawMatches(image, keypoints_sift, image_scaled, keypoints_sift_copy, good_matches_sift, img_matches_sift);
    cv::drawMatches(image, keypoints_sgloh2, image_scaled, keypoints_sgloh2_copy, good_matches_sgloh2, img_matches_sgloh2);

    cv::imshow("Good Matches SIFT", img_matches_sift);
    cv::imshow("Good Matches sGLOH2", img_matches_sgloh2);
    cv::waitKey(0);
}


void processImage6(const std::string& imageFileName) {
    // Load the image in grayscale
    cv::Mat image = cv::imread(imageFileName, cv::IMREAD_GRAYSCALE);
    cv::Mat image_rotated;

    // Rotate the image by 45 degrees
    cv::Point2f center((image.cols-1)/2.0, (image.rows-1)/2.0);
    cv::Mat rot = cv::getRotationMatrix2D(center, 45, 1.0);
    cv::warpAffine(image, image_rotated, rot, image.size());

    // Initialize SIFT detector and compute keypoints and descriptors for both images
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints_sift, keypoints_sift_copy;
    cv::Mat descriptors_sift, descriptors_sift_copy;
    sift->detectAndCompute(image, cv::noArray(), keypoints_sift, descriptors_sift);
    sift->detectAndCompute(image_rotated, cv::noArray(), keypoints_sift_copy, descriptors_sift_copy);

    // Match the SIFT descriptors using Brute-Force matcher
    cv::BFMatcher matcher_sift;
    std::vector<cv::DMatch> matches_sift;
    matcher_sift.match(descriptors_sift, descriptors_sift_copy, matches_sift);

    // Filter out good matches based on their distance
    std::vector<cv::DMatch> good_matches_sift;
    double min_dist_sift = 100;
    for( int i = 0; i < descriptors_sift.rows; i++ ) {
        if( matches_sift[i].distance <= std::max(2*min_dist_sift, 0.02) ) {
            good_matches_sift.push_back(matches_sift[i]);
        }
    }

    // Initialize sGLOH2 descriptor and compute keypoints and descriptors for both images
    sGLOH2 sgloh2;
    std::vector<cv::KeyPoint> keypoints_sgloh2, keypoints_sgloh2_copy;
    cv::Mat descriptors_sgloh2, descriptors_sgloh2_copy;
    sgloh2.compute(image, keypoints_sgloh2, descriptors_sgloh2);
    sgloh2.compute(image_rotated, keypoints_sgloh2_copy, descriptors_sgloh2_copy);

    // Match the sGLOH2 descriptors using Brute-Force matcher
    cv::BFMatcher matcher_sgloh2(cv::NORM_L2);
    std::vector<cv::DMatch> matches_sgloh2;
    matcher_sgloh2.match(descriptors_sgloh2, descriptors_sgloh2_copy, matches_sgloh2);

    // Filter out good matches based on their distance
    std::vector<cv::DMatch> good_matches_sgloh2;
    double min_dist_sgloh2 = sGLOH_THRESHOLD;
    for( int i = 0; i < descriptors_sgloh2.rows; i++ ) {
        if( matches_sgloh2[i].distance <= std::max(2*min_dist_sgloh2, 0.02) ) {
            good_matches_sgloh2.push_back(matches_sgloh2[i]);
        }
    }

    // Print the number of good matches for both SIFT and sGLOH2
    std::cout << "Number of good matches with sGLOH2: " << good_matches_sgloh2.size() << std::endl;
    std::cout << "Number of good matches with SIFT: " << good_matches_sift.size() << std::endl;

    // Draw the good matches for both SIFT
    cv::Mat img_matches_sift, img_matches_sgloh2;
    cv::drawMatches(image, keypoints_sift, image_rotated, keypoints_sift_copy, good_matches_sift, img_matches_sift);
    cv::drawMatches(image, keypoints_sgloh2, image_rotated, keypoints_sgloh2_copy, good_matches_sgloh2, img_matches_sgloh2);

    cv::imshow("Good Matches SIFT", img_matches_sift);
    cv::imshow("Good Matches sGLOH2", img_matches_sgloh2);
    cv::waitKey(0);
}


#endif //SGLOH_OPENCV_TESTS_HPP
