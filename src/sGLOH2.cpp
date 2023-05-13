//
// Created by user on 5/10/23.
//

#include "sGLOH2.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

sGLOH2::sGLOH2(int m) : m(m) {
    // Initialize any other member variables as needed
}

// Compute using grid
void sGLOH2::compute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    // Define grid size
    int gridSize = 32;  // adjust this value according to your requirements

    // Generate keypoints in a grid
    keypoints.clear();
    for (int y = gridSize / 2; y < image.rows; y += gridSize) {
        for (int x = gridSize / 2; x < image.cols; x += gridSize) {
            keypoints.push_back(cv::KeyPoint(cv::Point2f(x, y), gridSize));
        }
    }

    // Define the size of the patch to extract around each keypoint
    int patchSize = gridSize;  // Patch size should match grid size

    // Compute descriptors for each keypoint
    for (auto& keypoint : keypoints) {
        // Calculate the top-left corner of the patch
        cv::Point2f topLeft(keypoint.pt.x - patchSize / 2, keypoint.pt.y - patchSize / 2);

        // Check if the patch is within the image boundaries
        if (topLeft.x < 0 || topLeft.y < 0 || topLeft.x + patchSize > image.cols || topLeft.y + patchSize > image.rows) {
            continue;  // Skip this keypoint if the patch is not fully within the image
        }

        // Extract the patch around the keypoint
        cv::Mat patch = image(cv::Rect(topLeft.x, topLeft.y, patchSize, patchSize));

        // Compute sGLOH descriptor for the patch
        cv::Mat descriptor = compute_sGLOH(patch);

        // Append descriptor to descriptors matrix
        descriptors.push_back(descriptor);
    }
}

cv::Mat sGLOH2::compute_sGLOH_single(const cv::Mat& patch) {
    cv::Mat descriptor = cv::Mat::zeros(1, 0, CV_32F);

    // Compute the size of the regions
    int region_size = patch.rows / m;

    // Define the bin edges for the histogram
    std::vector<float> binEdges = {0, 90, 180, 270, 360};  // Fill this with the desired bin edges

    // Compute the histogram for each region
    for (int r = 0; r < m; r++) {
        for (int d = 0; d < m; d++) {
            // Compute the region
            cv::Rect region(r * region_size, d * region_size, region_size, region_size);

            // Compute the histogram for the region
            cv::Mat histogramMat = computeCustomHistogram(patch(region), binEdges);

            // Concatenate the histogram to the descriptor.
            cv::hconcat(descriptor, histogramMat, descriptor);
        }
    }

    return descriptor;
}

cv::Mat sGLOH2::compute_sGLOH(const cv::Mat& patch) {
    // Compute the first sGLOH descriptor on the original patch
    cv::Mat descriptor1 = compute_sGLOH_single(patch);

    // Rotate the patch by pi/m degrees
    cv::Mat patch_rotated;
    cv::Point2f center(patch.cols/2.0, patch.rows/2.0);
    double angle = 180.0 / m;
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
    // try cv::INTER_NEAREST and cv::INTER_CUBIC default is cv::INTER_LINEAR
    cv::warpAffine(patch, patch_rotated, rotation_matrix, patch.size(), cv::INTER_NEAREST);

    // Compute the second sGLOH descriptor on the rotated patch
    cv::Mat descriptor2 = compute_sGLOH_single(patch_rotated);

    // Concatenate the two descriptors
    cv::Mat descriptor;
    cv::hconcat(descriptor1, descriptor2, descriptor);

    // Normalize the descriptor
    cv::normalize(descriptor, descriptor, 1, 0, cv::NORM_L1);

    return descriptor;
}

double sGLOH2::distance(const cv::Mat& H_star1, const cv::Mat& H_star2) {
    double min_distance = std::numeric_limits<double>::max();

    // Separate the two descriptors in H_star1 and H_star2
    cv::Mat H1_1 = H_star1(cv::Range(0, 1), cv::Range(0, H_star1.cols/2));
    cv::Mat H1_2 = H_star1(cv::Range(0, 1), cv::Range(H_star1.cols/2, H_star1.cols));
    cv::Mat H2_1 = H_star2(cv::Range(0, 1), cv::Range(0, H_star2.cols/2));
    cv::Mat H2_2 = H_star2(cv::Range(0, 1), cv::Range(H_star2.cols/2, H_star2.cols));

    // Compute the cyclic shifts of H2_1 and H2_2
    for(int k = 0; k < m; ++k) {
        // Shift H2_1 and H2_2
        cv::Mat H2_1_shifted = cyclicShift(H2_1, k);
        cv::Mat H2_2_shifted = cyclicShift(H2_2, k);

        // Concatenate the shifted descriptors
        cv::Mat H2_shifted;
        cv::hconcat(H2_1_shifted, H2_2_shifted, H2_shifted);

        // Compute the distance to H_star1
        double distance = cv::norm(H_star1, H2_shifted, cv::NORM_L2);

        // Update the minimum distance
        if (distance < min_distance) {
            min_distance = distance;
        }
    }

    return min_distance;
}

cv::Mat sGLOH2::cyclicShift(const cv::Mat& descriptor, int k) {
    cv::Mat shifted_descriptor = descriptor.clone();

    // Compute the size of the blocks
    int block_size = descriptor.cols / m;

    // Shift each block by k positions
    for (int i = 0; i < m; ++i) {
        cv::Mat block = descriptor(cv::Range(0, 1), cv::Range(i * block_size, (i + 1) * block_size));

        // Convert the block to a vector
        std::vector<float> block_vec;
        block.copyTo(block_vec);

        // Rotate the block vector by k positions
        std::rotate(block_vec.begin(), block_vec.begin() + k, block_vec.end());

        // Convert the vector back to a Mat and replace the block in the descriptor
        cv::Mat shifted_block = cv::Mat(block_vec).reshape(1, 1);
        shifted_block.copyTo(shifted_descriptor(cv::Range(0, 1), cv::Range(i * block_size, (i + 1) * block_size)));
    }

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

// Use radians to compute the histogram
cv::Mat sGLOH2::computeHistogram(const cv::Mat& region, int m) {
    // Compute the gradient of the region
    cv::Mat grad_x, grad_y;
    cv::Sobel(region, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(region, grad_y, CV_32F, 0, 1, 3);

    // Compute the magnitude and orientation of the gradient
    cv::Mat magnitude, orientation;
    cv::cartToPolar(grad_x, grad_y, magnitude, orientation, true);

    // Convert orientation from radians to degrees
    orientation = orientation * 180.0 / CV_PI;

    // Compute the histogram of orientations
    cv::Mat histogram;
    int histSize = m;  // Number of bins
    float range[] = { 0, 360 } ;  // Orientation ranges from 0 to 360 degrees
    const float* histRange = { range };
    cv::calcHist(&orientation, 1, 0, cv::Mat(), histogram, 1, &histSize, &histRange);

    // Reshape the histogram to a single-row matrix
    histogram = histogram.reshape(1, 1);

    return histogram;
}

////uses old Histogram
//cv::Mat sGLOH2::compute_sGLOH_single(const cv::Mat& patch) {
//    cv::Mat descriptor = cv::Mat::zeros(1, 0, CV_32F);
//
//    // Compute the size of the regions
//    int region_size = patch.rows / m;
//
//    // Compute the histogram for each region
//    for (int r = 0; r < m; r++) {
//        for (int d = 0; d < m; d++) {
//            // Compute the region
//            cv::Rect region(r * region_size, d * region_size, region_size, region_size);
//
//            // Compute the histogram for the region
//            cv::Mat histogram = computeHistogram(patch(region), m);
//
//            // Reshape the histogram to a single-row matrix.
//            histogram = histogram.reshape(1, 1);
//
//            // Concatenate the histogram to the descriptor.
//            cv::hconcat(descriptor, histogram, descriptor);
//        }
//    }
//
//    return descriptor;
//}

//cv::Mat sGLOH2::computeHistogram(const cv::Mat& region, int m) {
//    // Compute the gradient of the region
//    cv::Mat grad_x, grad_y;
//    cv::Sobel(region, grad_x, CV_32F, 1, 0, 3);
//    cv::Sobel(region, grad_y, CV_32F, 0, 1, 3);
//
//    // Compute the magnitude and orientation of the gradient
//    cv::Mat magnitude, orientation;
//    cv::cartToPolar(grad_x, grad_y, magnitude, orientation, true);
//
//    // Compute the histogram of orientations
//    cv::Mat histogram;
//    int histSize = m;  // Number of bins
//    float range[] = { 0, 360 } ;  // Orientation ranges from 0 to 360 degrees
//    const float* histRange = { range };
//    cv::calcHist(&orientation, 1, 0, cv::Mat(), histogram, 1, &histSize, &histRange);
//
//    // Reshape the histogram to a single-row matrix
//    histogram = histogram.reshape(1, 1);
//
//    return histogram;
//}

// Compute using SIFT detector
//void sGLOH2::compute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
//    // Create a SIFT detector
//    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
//
//    // Detect keypoints
//    detector->detect(image, keypoints);
//
//    // Compute descriptors for each keypoint
//    for (auto& keypoint : keypoints) {
//        // Define the size of the patch to extract around each keypoint
//        int patchSize = 32;  // We  may need to adjust this value
//
//        // Calculate the top-left corner of the patch
//        cv::Point2f topLeft(keypoint.pt.x - patchSize / 2, keypoint.pt.y - patchSize / 2);
//
//        // Check if the patch is within the image boundaries
//        if (topLeft.x < 0 || topLeft.y < 0 || topLeft.x + patchSize > image.cols || topLeft.y + patchSize > image.rows) {
//            continue;  // Skip this keypoint if the patch is not fully within the image
//        }
//
//        // Extract the patch around the keypoint
//        cv::Mat patch = image(cv::Rect(topLeft.x, topLeft.y, patchSize, patchSize));
//
//        // Compute sGLOH descriptor for the patch
//        cv::Mat descriptor = compute_sGLOH(patch);
//
//        // Append descriptor to descriptors matrix
//        descriptors.push_back(descriptor);
//    }
//}
