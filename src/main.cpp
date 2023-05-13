#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include "sGLOH2.hpp"

int main(int argc, char** argv) {

    // Load the image in grayscale
    cv::Mat image = cv::imread("Yennefer.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat image_flipped;

    // Flip the image
    //cv::flip(image, image_flipped, 1);

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
    sGLOH2 sgloh2(4);
    std::vector<cv::KeyPoint> keypoints_sgloh2, keypoints_sgloh2_flipped;
    cv::Mat descriptors_sgloh2, descriptors_sgloh2_flipped;
    sgloh2.compute(image, keypoints_sgloh2, descriptors_sgloh2);
    sgloh2.compute(image_flipped, keypoints_sgloh2_flipped, descriptors_sgloh2_flipped);

    // Match the sGLOH2 descriptors using Brute-Force matcher
    cv::BFMatcher matcher_sgloh2(cv::NORM_L2);
    std::vector<cv::DMatch> matches_sgloh2;
    matcher_sgloh2.match(descriptors_sgloh2, descriptors_sgloh2_flipped, matches_sgloh2);

    // Filter out good matches based on their distance
    std::vector<cv::DMatch> good_matches_sgloh2;
    double min_dist_sgloh2 = 100;
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

    return 0;
}








//#include "sGLOH2.hpp"
//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <algorithm>  // To use std::sort
//
//int main() {
//    // Load an image
//    cv::Mat image = cv::imread("sign.jpg", cv::IMREAD_GRAYSCALE);
//    if (image.empty()) {
//        std::cout << "Could not open or find the image" << std::endl;
//        return -1;
//    }
//
//    // Rotate the image
//    cv::Mat rotated_image;
//    cv::rotate(image, rotated_image, cv::ROTATE_180);  // 180 degree rotation
//
//    // Create a sGLOH2 object
//    sGLOH2 sgloh(4);  // You can adjust the parameter as needed
//
//    // Compute keypoints and descriptors for the original image
//    std::vector<cv::KeyPoint> keypoints;
//    cv::Mat descriptors;
//    sgloh.compute(image, keypoints, descriptors);
//
//    // Compute keypoints and descriptors for the rotated image
//    std::vector<cv::KeyPoint> rotated_keypoints;
//    cv::Mat rotated_descriptors;
//    sgloh.compute(rotated_image, rotated_keypoints, rotated_descriptors);
//
//    // Match descriptors using Brute-Force matcher
//    cv::BFMatcher matcher(cv::NORM_L2);
//    std::vector<cv::DMatch> matches;
//    matcher.match(descriptors, rotated_descriptors, matches);
//
//    // Sort matches by distance (smaller distance means better match)
//    std::sort(matches.begin(), matches.end());
//
//    // Keep only the top 15 matches
//    if (matches.size() > 15) {
//        matches.resize(15);
//    }
//
//    // Draw matches between the original image and the rotated image
//    cv::Mat imageMatches;
//    cv::drawMatches(image, keypoints, rotated_image, rotated_keypoints, matches, imageMatches);
//
//    // Display the image with matches
//    cv::namedWindow("Matches", cv::WINDOW_NORMAL);
//    cv::imshow("Matches", imageMatches);
//    cv::waitKey(0);
//
//    return 0;
//}
//
//
//
