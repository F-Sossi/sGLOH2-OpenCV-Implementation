//
// Created by user on 5/27/23.
//

#ifndef SGLOH_OPENCV_SGLOHUTILITES_HPP
#define SGLOH_OPENCV_SGLOHUTILITES_HPP

#include "sGLOH2.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

/**
 * @brief Matches sGLOH descriptors using block matching.
 *
 * @param query The query descriptor.
 * @param descriptors The set of descriptors to match against.
 * @param k The number of nearest neighbors to return.
 * @param indices The indices of the k nearest neighbors.
 * @param distances The distances to the k nearest neighbors.
 *
 * This method matches the query descriptor against the set of descriptors using block matching.
 * It divides the sGLOH descriptor into blocks and compares the blocks between the query descriptor
 * and the candidate descriptors. It then computes the sum of the distance vectors for each candidate
 * descriptor and returns the top k candidates.
 */
void match_blocks(const cv::Mat& query, const cv::Mat& descriptors, int k, std::vector<int>& indices, std::vector<float>& distances) {
    // Divide the sGLOH descriptor into blocks.
    int num_blocks = 16;
    int block_size = 8;
    int num_bins = 128;
    cv::Mat query_blocks(num_blocks, block_size * num_bins, CV_32F);
    for (int i = 0; i < num_blocks; i++) {
        cv::Mat block = query(cv::Range(i * block_size, (i + 1) * block_size), cv::Range::all());
        cv::Mat block_hist;
        cv::normalize(block.reshape(1, 1), block_hist, 1, 0, cv::NORM_L2);
        block_hist.copyTo(query_blocks.row(i));
    }

    // Compute the Euclidean distance between the corresponding blocks in the query descriptor and the candidate descriptors.
    cv::Mat distances_mat(descriptors.rows, num_blocks, CV_32F);
    for (int i = 0; i < num_blocks; i++) {
        cv::Mat query_block = query_blocks.row(i);
        cv::Mat descriptors_block = descriptors(cv::Range::all(), cv::Range(i * block_size * num_bins, (i + 1) * block_size * num_bins));
        cv::Mat block_distances;
        cv::reduce((query_block - descriptors_block).mul(query_block - descriptors_block), block_distances, 1, cv::REDUCE_SUM);
        block_distances.copyTo(distances_mat.col(i));
    }

    // Compute the sum of the distance vectors for each candidate descriptor.
    cv::Mat distances_sum;
    cv::reduce(distances_mat, distances_sum, 1, cv::REDUCE_SUM);

    // Sort the candidate descriptors by their distance values and return the top k candidates.
    cv::Mat indices_mat;
    cv::sortIdx(distances_sum, indices_mat, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
    indices_mat.rowRange(0, k).copyTo(indices);
    distances_sum.rowRange(0, k).copyTo(distances);
}

/**
 * @brief Compares two images using sGLOH descriptors and block matching.
 *
 * @param img1 The first image.
 * @param img2 The second image.
 * @param k The number of nearest neighbors to return.
 * @param threshold The distance threshold for considering a match.
 * @return A vector of DMatch objects representing the matches between the two images.
 *
 * This method compares two images using sGLOH descriptors and block matching.
 * It first resizes the images to a fixed size of 512x512.
 * It then computes the sGLOH descriptors for the keypoints in both images.
 * It matches the descriptors using the `match_blocks` function and filters the matches
 * based on the distance threshold.
 */
std::vector<cv::DMatch> ImageCompareSgloh3(const cv::Mat& img1, const cv::Mat& img2, int knn, float ratio) {
    // Create sGLOH detector
    sGLOH2 sgloh;

    // Resize images
    cv::Mat resized1, resized2;
    cv::resize(img1, resized1, cv::Size(512, 512));
    cv::resize(img2, resized2, cv::Size(512, 512));

    // Detect keypoints and compute descriptors for both images
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    sgloh.compute(resized1, keypoints1, descriptors1);
    sgloh.compute(resized2, keypoints2, descriptors2);

    // Debug output
    std::cout << "descriptors1 size: " << descriptors1.size() << ", type: " << descriptors1.type() << std::endl;
    std::cout << "descriptors2 size: " << descriptors2.size() << ", type: " << descriptors2.type() << std::endl;

    //print out size of one descriptor
    std::cout << "descriptor size: " << sizeof(descriptors1.row(1)) << std::endl;

    // debug keypoints
    std::cout << "keypoints1 size: " << keypoints1.size() << std::endl;
    std::cout << "keypoints2 size: " << keypoints2.size() << std::endl;

    // print size of one keypoint
    std::cout << "keypoint size: " << sizeof(keypoints1[1]) << std::endl;

    // Create FLANN matcher
    cv::FlannBasedMatcher matcher;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, knn);

    // Debug output
    std::cout << "knn_matches size: " << knn_matches.size() << std::endl;

    // Filter matches using ratio test
    std::vector<cv::DMatch> matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio * knn_matches[i][1].distance) {
            matches.push_back(knn_matches[i][0]);
        }
    }

    // Debug output
    std::cout << "matches size: " << matches.size() << std::endl;

    return matches;
}

/**
 * @brief Compares two images using sGLOH descriptors and block matching.
 *
 * @param img1 The first image.
 * @param img2 The second image.
 * @param keypoints1 The keypoints detected in the first image.
 * @param keypoints2 The keypoints detected in the second image.
 * @param k The number of nearest neighbors to return.
 * @param threshold The distance threshold for considering a match.
 * @return A vector of DMatch objects representing the matches between the two images.
 *
 * This method compares two images using sGLOH descriptors and block matching.
 * It first resizes the images to a fixed size of 512x512.
 * It then computes the sGLOH descriptors for the keypoints in both images.
 * It matches the descriptors using the `match_blocks` function and filters the matches
 * based on the distance threshold.
 */
std::vector<cv::DMatch> ImageCompareSgloh4(const cv::Mat& img1, const cv::Mat& img2, std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2, int knn, float ratio) {
    // Create sGLOH detector
    sGLOH2 sgloh;

    // Resize images
    cv::Mat resized1, resized2;
    cv::resize(img1, resized1, cv::Size(512, 512));
    cv::resize(img2, resized2, cv::Size(512, 512));

    // Compute descriptors for both images
    cv::Mat descriptors1, descriptors2;
    sgloh.compute(resized1, keypoints1, descriptors1);
    sgloh.compute(resized2, keypoints2, descriptors2);

    // Debug output
    std::cout << "descriptors1 size: " << descriptors1.size() << ", type: " << descriptors1.type() << std::endl;
    std::cout << "descriptors2 size: " << descriptors2.size() << ", type: " << descriptors2.type() << std::endl;

    //print out size of one descriptor
    std::cout << "descriptor size: " << sizeof(descriptors1.row(1)) << std::endl;

    // debug keypoints
    std::cout << "keypoints1 size: " << keypoints1.size() << std::endl;
    std::cout << "keypoints2 size: " << keypoints2.size() << std::endl;

    // print size of one keypoint
    std::cout << "keypoint size: " << sizeof(keypoints1[1]) << std::endl;

    // Create FLANN matcher
    cv::FlannBasedMatcher matcher;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, knn);

    // Debug output
    std::cout << "knn_matches size: " << knn_matches.size() << std::endl;

    // Filter matches using ratio test
    std::vector<cv::DMatch> matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio * knn_matches[i][1].distance) {
            matches.push_back(knn_matches[i][0]);
        }
    }

    // Debug output
    std::cout << "matches size: " << matches.size() << std::endl;

    return matches;
}

#endif //SGLOH_OPENCV_SGLOHUTILITES_HPP
