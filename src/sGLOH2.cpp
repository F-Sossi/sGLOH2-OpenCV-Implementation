//
// Created by user on 5/10/23.
//

#include "sGLOH2.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// good sizes 31, 41, 51, 61, 71, 81, 91, 101, 111, 121
constexpr int PATCH_SIZE = 81;

sGLOH2::sGLOH2(int m) : m(m) {
    // Initialize any other member variables as needed
}

void sGLOH2::compute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {

//    // Convert the image to grayscale if it isn't already
//    cv::Mat grayImage;
//    if (image.channels() == 1) {
//        grayImage = image;
//    } else {
//        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
//    }
//
//    // Simplify the image using Gaussian blur
//    cv::Mat blurredImage;
//    cv::GaussianBlur(grayImage, blurredImage, cv::Size(5, 5), 0, 0);
//
//    // Perform Canny edge detection
//    cv::Mat edges;
//    cv::Canny(blurredImage, edges, 100, 200);  // Threshold values may need to be adjusted for your specific use case


    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
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
        cv::normalize(descriptor, descriptor);

        descriptors.push_back(descriptor);
    }
}

cv::Mat sGLOH2::compute_sGLOH_single(const cv::Mat& patch) {
    // Initialize the descriptor as a single-row matrix of zeros.
    cv::Mat descriptor = cv::Mat::zeros(1, 0, CV_32F);

    // Define the center of the patch.
    int cx = patch.cols / 2;
    int cy = patch.rows / 2;

    // Compute the radius of the circle.
    int radius = std::min(patch.rows, patch.cols) / 2;

    // Compute the histogram for each quadrant.
    for (int quadrant = 0; quadrant < 4; quadrant++) {
        // Initialize a mask for the quadrant.
        cv::Mat quadrant_mask = cv::Mat::zeros(patch.size(), CV_8U);

        // Set the pixels within the quadrant to 1.
        for (int y = 0; y < patch.rows; y++) {
            for (int x = 0; x < patch.cols; x++) {
                int dx = x - cx;
                int dy = y - cy;
                int distance = std::sqrt(dx * dx + dy * dy);
                if (distance < radius) {
                    // Check which quadrant the pixel is in.
                    bool in_quadrant;
                    switch (quadrant) {
                        case 0:  // Top-right quadrant
                            in_quadrant = dx >= 0 && dy < 0;
                            break;
                        case 1:  // Bottom-right quadrant
                            in_quadrant = dx >= 0 && dy >= 0;
                            break;
                        case 2:  // Bottom-left quadrant
                            in_quadrant = dx < 0 && dy >= 0;
                            break;
                        case 3:  // Top-left quadrant
                            in_quadrant = dx < 0 && dy < 0;
                            break;
                    }
                    if (in_quadrant) {
                        quadrant_mask.at<uchar>(y, x) = 1;
                    }
                }
            }
        }

        // Compute the histogram for the quadrant.
        cv::Mat histogramMat = computeHistogram(patch, quadrant_mask, 8);

        // Concatenate the histogram to the descriptor.
        cv::hconcat(descriptor, histogramMat, descriptor);
    }

    // Return the final descriptor.
    return descriptor;
}



cv::Mat sGLOH2::compute_sGLOH(const cv::Mat& patch) {
    // The compute_sGLOH_single function computes the sGLOH descriptor for a given patch.
    // It divides the patch into m*m regions and computes a gradient orientation histogram for each region.
    // These histograms are then concatenated to form the final descriptor for the patch.
    cv::Mat descriptor1 = compute_sGLOH_single(patch);

    // To achieve rotation invariance, the patch is rotated by pi/m degrees.
    // The rotation is performed around the center of the patch.
    cv::Mat patch_rotated;
    cv::Point2f center(patch.cols/2.0, patch.rows/2.0);
    double angle = 180.0 / m;
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(patch, patch_rotated, rotation_matrix, patch.size(), cv::INTER_NEAREST);

    // The sGLOH descriptor is then computed for the rotated patch.
    // This is done in the same way as for the original patch.
    cv::Mat descriptor2 = compute_sGLOH_single(patch_rotated);

    // The descriptors for the original and rotated patches are then concatenated.
    // This forms the final sGLOH descriptor, which contains information from two orientations of the patch.
    cv::Mat descriptor;
    cv::hconcat(descriptor1, descriptor2, descriptor);

    // The final descriptor is then normalized to have a L1 norm of 1.
    // This makes the descriptor invariant to changes in the contrast of the patch.
    // The L1 norm is used because it is less sensitive to outliers than the L2 norm.
    cv::normalize(descriptor, descriptor, 1, 0, cv::NORM_L1);

    // The final, normalized sGLOH descriptor is then returned.
    return descriptor;
}

double sGLOH2::distance(const cv::Mat& H_star1, const cv::Mat& H_star2) {
    // Initialize the maximum similarity to the minimum possible value.
    double max_similarity = -1.0;
    int best_rotation = 0;

    // Separate the two descriptors in H_star1 and H_star2 into two halves each.
    cv::Mat H1_1 = H_star1(cv::Range(0, 1), cv::Range(0, H_star1.cols/2));
    cv::Mat H1_2 = H_star1(cv::Range(0, 1), cv::Range(H_star1.cols/2, H_star1.cols));
    cv::Mat H2_1 = H_star2(cv::Range(0, 1), cv::Range(0, H_star2.cols/2));
    cv::Mat H2_2 = H_star2(cv::Range(0, 1), cv::Range(H_star2.cols/2, H_star2.cols));

    // Compute the cyclic shifts of H2_1 and H2_2.
    for(int k = 0; k < m; ++k) {
        // Shift H2_1 and H2_2 by k positions.
        cv::Mat H2_1_shifted = cyclicShift(H2_1, k);
        cv::Mat H2_2_shifted = cyclicShift(H2_2, k);

        // Concatenate the shifted descriptors to form the complete descriptor.
        cv::Mat H2_shifted;
        cv::hconcat(H2_1_shifted, H2_2_shifted, H2_shifted);

        // Compute the similarity between H_star1 and the shifted H2.
        double similarity = cosine_similarity(H_star1, H2_shifted);

        // Update the maximum similarity and best rotation if the current similarity is larger.
        if (similarity > max_similarity) {
            max_similarity = similarity;
            best_rotation = k;
        }
    }

    // Reject incongruous correspondences by adding a global constraint on the rotations.
    int rotation_threshold = 5;  // Adjust this value according to your requirements
    if (abs(best_rotation - m/2) > rotation_threshold) {
        return -1.0;
    }

    // Return the maximum similarity found.
    return max_similarity;
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
    // The descriptor is divided into m blocks along the column dimension.
    int block_size = descriptor.cols / m;

    // Shift each block by k positions. This is done by converting each block to a vector,
    // rotating the vector, and then converting it back to a block.
    for (int i = 0; i < m; ++i) {
        // Extract the i-th block from the descriptor.
        cv::Mat block = descriptor(cv::Range(0, 1), cv::Range(i * block_size, (i + 1) * block_size));

        // Convert the block to a vector. This is done to facilitate the rotation operation.
        std::vector<float> block_vec;
        block.copyTo(block_vec);

        // Rotate the block vector by k positions. The rotate function modifies the vector in-place.
        std::rotate(block_vec.begin(), block_vec.begin() + k, block_vec.end());

        // Convert the rotated vector back to a Mat and replace the i-th block in the descriptor.
        cv::Mat shifted_block = cv::Mat(block_vec).reshape(1, 1);
        shifted_block.copyTo(shifted_descriptor(cv::Range(0, 1), cv::Range(i * block_size, (i + 1) * block_size)));
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

    // Compute the histogram of orientations.
    cv::Mat histogram;
    int histSize = m;  // Number of bins.
    float range[] = { 0, 360 };  // Orientation ranges from 0 to 360 degrees.
    const float* histRange = { range };
    cv::calcHist(&orientation, 1, 0, mask, histogram, 1, &histSize, &histRange);

    // Reshape the histogram to a single-row matrix.
    histogram = histogram.reshape(1, 1);

    return histogram;
}


