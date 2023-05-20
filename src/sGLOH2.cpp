//
// Created by user on 5/10/23.
//

#include "sGLOH2.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// Only use odd sizes for the patch size (circular region of interest)
// good sizes 31, 41, 51, 61, 71, 81, 91, 101, 111, 121
constexpr int PATCH_SIZE = 61;
constexpr int N = 2;
constexpr int M = 8;
constexpr int Q = 8;

sGLOH2::sGLOH2(int m) : m(m) {
    // Initialize any other member variables as needed
}

//void sGLOH2::compute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
//
////    // Did some image preprocessing here it did not have a noticeable effect on the results
////    // Convert the image to grayscale if it is not already
////    cv::Mat grayImage;
////    if (image.channels() == 1) {
////        grayImage = image;
////    } else {
////        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
////    }
////
////    // Simplify the image using Gaussian blur
////    cv::Mat blurredImage;
////    cv::GaussianBlur(grayImage, blurredImage, cv::Size(5, 5), 0, 0);
////
////    // Perform Canny edge detection
////    cv::Mat edges;
////    cv::Canny(blurredImage, edges, 100, 200);  // Threshold values may need to be adjusted for your specific use case
//
//
//    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
//    detector->detect(image, keypoints);
//
//    int patchSize = PATCH_SIZE;
//
//    for (auto& keypoint : keypoints) {
//        cv::Point2f topLeft(keypoint.pt.x - patchSize / 2, keypoint.pt.y - patchSize / 2);
//
//        if (topLeft.x < 0 || topLeft.y < 0 || topLeft.x + patchSize > image.cols || topLeft.y + patchSize > image.rows) {
//            continue;
//        }
//
//        cv::Mat patch = image(cv::Rect(topLeft.x, topLeft.y, patchSize, patchSize));
//
//        // Create a copy of the patch to be rotated.
//        cv::Mat rotatedPatch = patch.clone();
//
////        // Create a rotation matrix for rotating the patch to align with the keypoint orientation.
////        cv::Mat rotMat = cv::getRotationMatrix2D(cv::Point2f(patchSize / 2, patchSize / 2), -keypoint.angle, 1.0);
////
////        // Apply the rotation to the copy of the patch.
////        cv::warpAffine(rotatedPatch, rotatedPatch, rotMat, rotatedPatch.size());
//
//        cv::Mat descriptor = compute_sGLOH(rotatedPatch);
//
//        // Normalize the descriptor.
//        cv::normalize(descriptor, descriptor, 1, 0, cv::NORM_L2);
//
//        descriptors.push_back(descriptor);
//    }
//}

void sGLOH2::compute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {

    // Create ORB detector
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    detector->detect(image, keypoints);

    int patchSize = PATCH_SIZE;

    for (auto& keypoint : keypoints) {
        cv::Point2f topLeft(keypoint.pt.x - patchSize / 2, keypoint.pt.y - patchSize / 2);

        if (topLeft.x < 0 || topLeft.y < 0 || topLeft.x + patchSize > image.cols || topLeft.y + patchSize > image.rows) {
            continue;
        }

        cv::Mat patch = image(cv::Rect(topLeft.x, topLeft.y, patchSize, patchSize));

        // Create a copy of the patch to be rotated.
        cv::Mat rotatedPatch = patch.clone();

        // Create a rotation matrix for rotating the patch to align with the keypoint orientation.
        cv::Mat rotMat = cv::getRotationMatrix2D(cv::Point2f(patchSize / 2, patchSize / 2), -keypoint.angle, 1.0);

        // Apply the rotation to the copy of the patch.
        cv::warpAffine(rotatedPatch, rotatedPatch, rotMat, rotatedPatch.size());

        cv::Mat descriptor = compute_sGLOH(rotatedPatch);

        // Normalize the descriptor.
        cv::normalize(descriptor, descriptor, 1, 0, cv::NORM_L2);

        descriptors.push_back(descriptor);
    }
}


cv::Mat sGLOH2::compute_sGLOH_single(const cv::Mat& patch) {
    // Initialize the descriptor.
    cv::Mat descriptor = cv::Mat::zeros(1, Q * N * M, CV_32F);

    // Define the center of the patch.
    int cx = patch.cols / 2;
    int cy = patch.rows / 2;

    // Compute the radius of the circle.
    int radius = std::min(patch.rows, patch.cols) / 2;

    // Compute the histogram for each sector in each ring.
    for (int ring = 0; ring < N; ring++) {
        for (int sector = 0; sector < M; sector++) {
            // Initialize a mask for the region.
            cv::Mat region_mask = cv::Mat::zeros(patch.size(), CV_8U);

            // Set the pixels within the region to 1.
            for (int y = 0; y < patch.rows; y++) {
                for (int x = 0; x < patch.cols; x++) {
                    int dx = x - cx;
                    int dy = y - cy;
                    float angle = std::atan2(dy, dx);
                    int distance = std::sqrt(dx * dx + dy * dy);

                    // Check which ring and sector the pixel is in.
                    bool in_ring = (ring * radius / N <= distance) && (distance < (ring + 1) * radius / N);
                    bool in_sector = (sector * 2 * CV_PI / M <= angle) && (angle < (sector + 1) * 2 * CV_PI / M);

                    if (in_ring && in_sector) {
                        region_mask.at<uchar>(y, x) = 1;
                    }
                }
            }

            // Compute the histogram for the region.
            cv::Mat histogramMat = computeHistogram(patch, region_mask, Q);

            // Concatenate the histogram to the descriptor.
            cv::hconcat(descriptor, histogramMat, descriptor);
        }
    }

    // Normalize the descriptor.
    cv::normalize(descriptor, descriptor, 1, 0, cv::NORM_L1);

    // Return the final descriptor.
    return descriptor;
}

cv::Mat sGLOH2::compute_sGLOH(const cv::Mat& patch) {
    // The compute_sGLOH function computes the sGLOH2 descriptor for a given patch.
    // It divides the patch into m*m regions and computes a gradient orientation histogram for each region.
    // These histograms are then concatenated to form the final descriptor for the patch.

    cv::Mat descriptor;
    cv::Point2f center(patch.cols/2.0, patch.rows/2.0);
    double angle_step = 360.0 / (2*M); // angle in degrees

    for(int i = 0; i < 2*M; i++) {
        cv::Mat patch_rotated;
        double angle = i * angle_step;

        cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::warpAffine(patch, patch_rotated, rotation_matrix, patch.size(), cv::INTER_NEAREST);

        cv::Mat descriptor_part = compute_sGLOH_single(patch_rotated);

        if (descriptor.empty()) {
            descriptor = descriptor_part;
        } else {
            cv::hconcat(descriptor, descriptor_part, descriptor);
        }
    }

    // Normalize the descriptor.
    cv::normalize(descriptor, descriptor, 1, 0, cv::NORM_L2);

    // The final, normalized sGLOH2 descriptor is then returned.
    return descriptor;
}


double sGLOH2::distance(const cv::Mat& H_star1, const cv::Mat& H_star2) {
    // Initialize the minimum distance to the maximum possible value.
    double min_distance = std::numeric_limits<double>::max();
    int best_rotation = 0;

    // Split each descriptor into two halves.
    cv::Mat H1_1 = H_star1(cv::Range(0, 1), cv::Range(0, H_star1.cols / 2));
    cv::Mat H1_2 = H_star1(cv::Range(0, 1), cv::Range(H_star1.cols / 2, H_star1.cols));
    cv::Mat H2_1, H2_2;

    // Try each rotation for the second descriptor.
    for (int k = 0; k < M; ++k) {
        H2_1 = cyclicShift(H_star2(cv::Range(0, 1), cv::Range(0, H_star2.cols / 2)), k);
        H2_2 = cyclicShift(H_star2(cv::Range(0, 1), cv::Range(H_star2.cols / 2, H_star2.cols)), k);

        // Concatenate the halves into a full descriptor for this rotation.
        cv::Mat H2_shifted;
        cv::hconcat(H2_1, H2_2, H2_shifted);

        // Compute the distance between H_star1 and the rotated H2.
        double distance = cv::norm(H_star1, H2_shifted, cv::NORM_L2);

        // Update the minimum distance and best rotation if the current distance is smaller.
        if (distance < min_distance) {
            min_distance = distance;
            best_rotation = k;
        }
    }

    // Add a global constraint on the rotations.
    int rotation_threshold = M / 2;  // Adjust this value according to your requirements
    if (abs(best_rotation - M / 2) > rotation_threshold) {
        return std::numeric_limits<double>::max();
    }

    // Return the minimum distance. The lower the distance, the better the match.
    return min_distance;
}

double sGLOH2::cosine_similarity(const cv::Mat& H1, const cv::Mat& H2) {
    // Convert the descriptors to 1D vectors.
    cv::Mat H1_vec = H1.reshape(1, 1);
    cv::Mat H2_vec = H2.reshape(1, 1);

    // Compute the dot product of the vectors.
    double dot = H1_vec.dot(H2_vec);

    // Compute the L2 norms of the vectors.
    double norm_H1 = cv::norm(H1_vec, cv::NORM_L2);
    double norm_H2 = cv::norm(H2_vec, cv::NORM_L2);

    // Compute the cosine similarity.
    double cos_sim = dot / (norm_H1 * norm_H2);

    // Return the cosine similarity.
    return cos_sim;
}

cv::Mat sGLOH2::cyclicShift(const cv::Mat& descriptor, int k) {
    // Clone the descriptor to create a new matrix that will be modified and returned.
    cv::Mat shifted_descriptor = descriptor.clone();

    // Compute the size of the blocks. Each block corresponds to a region of the image.
    // The descriptor is divided into 2*m blocks along the column dimension, as it's a concatenation of two sGLOH descriptors.
    int block_size = descriptor.cols / (2*M);

    // Shift each block by k positions. This is done by converting each block to a vector,
    // rotating the vector, and then converting it back to a block.
    for (int i = 0; i < 2*M; ++i) {
        // Extract the i-th block from the descriptor.
        cv::Mat block = descriptor(cv::Range::all(), cv::Range(i * block_size, (i + 1) * block_size));

        // Convert the block to a vector. This is done to facilitate the rotation operation.
        std::vector<float> block_vec;
        block.reshape(1, 1).copyTo(block_vec);

        // Rotate the block vector by k positions. The rotate function modifies the vector in-place.
        std::rotate(block_vec.begin(), block_vec.begin() + block_vec.size() - k % block_vec.size(), block_vec.end());

        // Convert the rotated vector back to a Mat and replace the i-th block in the shifted_descriptor.
        cv::Mat(block_vec).reshape(1, block.rows).copyTo(shifted_descriptor(cv::Range::all(), cv::Range(i * block_size, (i + 1) * block_size)));
    }

    // Return the descriptor with cyclically shifted blocks.
    return shifted_descriptor;
}

cv::Mat sGLOH2::computeCustomHistogram(const cv::Mat& data, const std::vector<float>& binEdges) {
    std::vector<int> histogram(binEdges.size() - 1, 0);
    for (int i = 0; i < data.rows; ++i) {
        for (int j = 0; j < data.cols; ++j) {
            float value = data.at<float>(i, j);

            // Find the bin that this value belongs to
            for (size_t bin = 0; bin < binEdges.size() - 1; ++bin) {
                if (value >= binEdges[bin] && value < binEdges[bin + 1]) {
                    ++histogram[bin];
                    break;
                }
            }
        }
    }

    // Convert the histogram to a cv::Mat and ensure it's a single-row matrix.
    cv::Mat histogramMat = cv::Mat(histogram).reshape(1, 1);

    // Ensure that the histogramMat has the same data type as the descriptor matrix
    histogramMat.convertTo(histogramMat, CV_32F);

    return histogramMat;
}

cv::Mat sGLOH2::computeHistogram(const cv::Mat& patch, const cv::Mat& mask, int m) {
    // Compute the gradient of the region.
    cv::Mat grad_x, grad_y;
    cv::Sobel(patch, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(patch, grad_y, CV_32F, 0, 1, 3);

    // Compute the magnitude and orientation of the gradient.
    cv::Mat magnitude, orientation;
    cv::cartToPolar(grad_x, grad_y, magnitude, orientation, true);

    // Convert orientation from radians to degrees.
    orientation = orientation * 180.0 / CV_PI;

    // Initialize histogram
    cv::Mat histogram = cv::Mat::zeros(1, m, CV_32F);

    // Compute weighted histogram manually
    for (int i = 0; i < orientation.rows; ++i) {
        for (int j = 0; j < orientation.cols; ++j) {
            if (mask.at<uchar>(i, j)) {
                int bin = cvRound(orientation.at<float>(i, j) / 360.0 * m) % m;
                histogram.at<float>(0, bin) += magnitude.at<float>(i, j);
            }
        }
    }

    // Reshape the histogram to a single-row matrix.
    histogram = histogram.reshape(1, 1);

    return histogram;
}



