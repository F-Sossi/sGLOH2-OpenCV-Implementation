/**
 * @file sGLOH2.cpp
 * @author Frank Sossi, Justin Boyer
 * @date 5/10/23
 * @brief Definition of the sGLOH2 class.
 *
 * This class provides the definition for the sGLOH2 class, including
 * constructors, computation and utility methods. The class is designed
 * for computing the sGLOH2 descriptor for a given image patch.
 *
 * Implemented functions:
 * sGLOH2::sGLOH2(int m = 8)
 * void sGLOH2::compute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
 * cv::Mat sGLOH2::compute_sGLOH(const cv::Mat& patch)
 * double sGLOH2::distance(const cv::Mat& H_star1, const cv::Mat& H_star2)
 * cv::Mat sGLOH2::cyclicShift(const cv::Mat &descriptor, int k)
 * cv::Mat sGLOH2::compute_sGLOH_single(const cv::Mat &patch)
 * cv::Mat sGLOH2::computeCustomHistogram(const cv::Mat &data, const std::vector<float> &binEdges)
 * cv::Mat sGLOH2::computeHistogram(const cv::Mat &patch, const cv::Mat &mask, int m)
 * double sGLOH2::cosine_similarity(const cv::Mat &H1, const cv::Mat &H2)
 *
 */

#include "sGLOH2.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

// Only use odd sizes for the patch size (circular region of interest)
// good sizes 31, 41, 51, 61, 71, 81, 91, 101, 111, 121 etc.
constexpr int PATCH_SIZE = 41;
constexpr int N = 2;
constexpr int M = 4;
constexpr int Q = 4;

/**
 * @brief Constructor for the sGLOH2 class. It precomputes masks for each ring and sector
 *        used in the computation of sGLOH descriptors.
 *
 * @param m The number of sectors in each ring of the sGLOH descriptor.
 *
 * This constructor creates a series of region masks which are then used to calculate
 * the descriptors of the input image. These masks correspond to different rings and
 * sectors of the patch surrounding the keypoint in an image, following the sGLOH descriptor
 * extraction methodology.
 *
 * Each mask is a binary image where the pixels belonging to a certain ring and sector
 * are set to 1, and the rest to 0.
 *
 * @note This constructor is computationally expensive as it involves multiple loops and
 *       trigonometric operations. The benefit of performing this computation in the
 *       constructor is that it only needs to be done once when an object of the sGLOH2
 *       class is created, and not every time when the compute method is called.
 */
sGLOH2::sGLOH2(int m) : m(m) {
    int patchSize = PATCH_SIZE;
    int radius = patchSize / 2;

    // Compute the masks for each ring and sector.
    region_masks.resize(N);
    for (int ring = 0; ring < N; ring++) {
        region_masks[ring].resize(M);
        for (int sector = 0; sector < M; sector++) {
            // Initialize a mask for the region.
            cv::Mat region_mask = cv::Mat::zeros(cv::Size(patchSize, patchSize), CV_8U);

            // Set the pixels within the region to 1.
            for (int y = 0; y < patchSize; y++) {
                for (int x = 0; x < patchSize; x++) {
                    int dx = x - patchSize / 2;
                    int dy = y - patchSize / 2;
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

            region_masks[ring][sector] = region_mask;
        }
    }
}

/**
 * @brief Computes the sGLOH descriptors for the given image and keypoints.
 *
 * @param image The input image.
 * @param keypoints The keypoints for which to compute the descriptors.
 * @param descriptors The computed descriptors.
 *
 * This method first uses an ORB detector to detect keypoints in the given image.
 * Then it creates a patch around each keypoint and aligns the patch with the keypoint's
 * orientation. This is achieved by rotating the patch around its center by the negative
 * angle of the keypoint.
 *
 * For each aligned patch, it computes the sGLOH descriptor by calling the compute_sGLOH
 * method, normalizes the descriptor, and stores it in the output matrix.
 *
 * The computation of descriptors is done in parallel for each keypoint to increase efficiency.
 * Finally, it concatenates the descriptors into a single matrix where each row corresponds to
 * a keypoint.
 *
 * @note The keypoints that are too close to the border of the image are skipped to avoid
 *       extracting patches outside of the image boundaries.
 */
void sGLOH2::compute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {

    // Create ORB detector
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    detector->detect(image, keypoints);

    int patchSize = PATCH_SIZE;

    std::vector<cv::Mat> descriptors_temp(keypoints.size());

    // Parallel loop across all keypoints
    cv::parallel_for_(cv::Range(0, keypoints.size()), [&](const cv::Range& range) {
        for (int r = range.start; r < range.end; r++) {
            auto& keypoint = keypoints[r];
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
            cv::normalize(descriptor, descriptor, 1, 0, cv::NORM_L1);

            descriptors_temp[r] = descriptor;

        }
    });

    // Concatenate descriptors
    cv::vconcat(descriptors_temp, descriptors);
}

/**
 * @brief Compute the Single GLOH (sGLOH) descriptor for a single image patch.
 *
 * This method computes the sGLOH descriptor by calculating the histogram for each sector in each ring
 * of the given image patch and then concatenating these histograms to form the final descriptor.
 * The histogram computation for each sector of each ring is done with the help of precomputed masks.
 * The descriptor is then normalized to ensure scale invariance.
 *
 * @param patch The image patch for which the sGLOH descriptor is to be computed. It must be a grayscale image.
 * @return The computed sGLOH descriptor as a single-row matrix of type CV_32F. The number of columns equals Q * N * M,
 *         where Q is the number of quantization levels, N is the number of rings, and M is the number of sectors.
 */
cv::Mat sGLOH2::compute_sGLOH_single(const cv::Mat& patch) {
    // Initialize the descriptor.
    cv::Mat descriptor = cv::Mat::zeros(1, Q * N * M, CV_32F);

    // Compute the histogram for each sector in each ring.
    for (int ring = 0; ring < N; ring++) {
        for (int sector = 0; sector < M; sector++) {
            // Use the precomputed mask for the region.
            cv::Mat region_mask = region_masks[ring][sector];

            // Compute the histogram for the region.
            cv::Mat histogramMat = computeHistogram(patch, region_mask, Q);

            // Place the histogram to the descriptor at the correct position.
            int start = (ring * M + sector) * Q;
            histogramMat.copyTo(descriptor.colRange(start, start + Q));
        }
    }

    // Normalize the descriptor.
    cv::normalize(descriptor, descriptor, 1, 0, cv::NORM_L1);

    // Return the final descriptor.
    return descriptor;
}

/**
 * @brief Computes the sGLOH2 descriptor for a given image patch.
 *
 * The method works by dividing the image patch into multiple regions, computing a gradient orientation histogram for each region,
 * and concatenating these histograms to form the final descriptor. This process is repeated for multiple orientations of the patch,
 * resulting in a rotation-invariant descriptor. Each orientation is obtained by rotating the patch around its center.
 *
 * The method uses the `compute_sGLOH_single` function to compute the descriptor for each orientation of the patch.
 *
 * After all descriptors have been computed, they are concatenated and normalized to form the final sGLOH2 descriptor.
 *
 * @param patch The image patch to be processed. It must be a single-channel matrix of type CV_8U.
 * @return The sGLOH2 descriptor for the patch as a single-row matrix of type CV_32F.
 */
cv::Mat sGLOH2::compute_sGLOH(const cv::Mat& patch) {
    // Compute the sGLOH descriptor for the original patch.
    cv::Mat descriptor = compute_sGLOH_single(patch);

    // Compute the center of the patch.
    cv::Point2f center(patch.cols/2.0, patch.rows/2.0);

    // Compute the rotation angle in degrees (pi/M converted to degrees).
    double angle_step = 360.0 / M;

    // Rotate the patch by the computed angle.
    cv::Mat patch_rotated;
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle_step, 1.0);
    cv::warpAffine(patch, patch_rotated, rotation_matrix, patch.size(), cv::INTER_NEAREST);

    // Compute the sGLOH descriptor for the rotated patch.
    cv::Mat descriptor_rotated = compute_sGLOH_single(patch_rotated);

    // Concatenate the original and rotated descriptors to form the final descriptor.
    cv::hconcat(descriptor, descriptor_rotated, descriptor);

    // Normalize the descriptor.
    cv::normalize(descriptor, descriptor, 1, 0, cv::NORM_L1);

    // Return the final descriptor.
    return descriptor;
}

/**
 * @brief Compute the extended sGLOH (sGLOH2) descriptor for a single image patch.
 *
 * This method computes the sGLOH2 descriptor by dividing the patch into m*m regions,
 * and for each region, a gradient orientation histogram is computed. This process is
 * then repeated for rotated versions of the patch, covering 2*M angles of rotation.
 * The histograms of all regions across all rotations are concatenated to form the final descriptor.
 *
 * The descriptor is then normalized to ensure scale invariance.
 *
 * @param patch The image patch for which the sGLOH2 descriptor is to be computed. It must be a grayscale image.
 * @return The computed sGLOH2 descriptor as a single-row matrix of type CV_32F. The number of columns is proportional
 *         to the number of sectors (M), number of rings (N), number of quantization levels (Q), and the number of rotation angles (2*M).
 */
double sGLOH2::distance(const cv::Mat& H_star1, const cv::Mat& H_star2) {
    // Initialize the minimum distance to the maximum possible value.
    double min_distance = std::numeric_limits<double>::max();
    int best_rotation = 0;

    cv::Mat H2_shifted;

    // Try each rotation for the second descriptor.
    for (int k = 0; k < M; ++k) {
        H2_shifted = cyclicShift(H_star2, 4);

        // Compute the distance between H_star1 and the rotated H2.
        double distance = cv::norm(H_star1, H2_shifted, cv::NORM_L1);

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

/**
 * @brief Apply a cyclic shift to the blocks of a sGLOH2 descriptor.
 *
 * This method applies a cyclic shift of k positions to each block of a sGLOH2 descriptor.
 * The blocks correspond to the m*m regions of the original image patch. The descriptor is
 * divided into 2*M blocks along the column dimension, as it's a concatenation of two sGLOH descriptors.
 * The shift is performed in-place on each block (converted to a vector for easy rotation),
 * and the modified descriptor is returned.
 *
 * @param descriptor The sGLOH2 descriptor to be shifted. It must be a single-row matrix of type CV_32F.
 * @param k The number of positions to shift the blocks in the descriptor. The direction of the shift is to the left.
 * @return The shifted descriptor as a single-row matrix of type CV_32F. The number of columns is the same as the input descriptor.
 */
cv::Mat sGLOH2::cyclicShift(const cv::Mat& descriptor, int k) {
    // Clone the descriptor to create a new matrix that will be modified and returned.
    cv::Mat shifted_descriptor = descriptor.clone();

    // Compute the size of the blocks. Each block corresponds to a region of the image.
    int block_size = 16; //Q * N * M / (2*M); // currently 4

    // Yes this is pointless for loop, but I wanted to preserve the ability to rotate in different ways
    for (int i = 0; i < 1; ++i) {
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

/**
 * @brief Compute a gradient orientation histogram for a given image patch and mask.
 *
 * This method first computes the gradient of the image patch, both in the x and y direction,
 * then computes the magnitude and orientation of the gradient. The orientation is converted from
 * radians to degrees. A histogram is then computed using these orientations, where each bin in
 * the histogram corresponds to a range of gradient orientations. The contribution of each pixel to a
 * histogram bin is weighted by the gradient magnitude at that pixel, and only pixels within the mask are considered.
 *
 * @param patch The image patch to be processed. It must be a single-channel matrix of type CV_8U.
 * @param mask The mask defining the region within the patch to consider when computing the histogram. It should be of the same size as the patch and of type CV_8U.
 * @param m The number of bins in the histogram.
 * @return The gradient orientation histogram as a single-row matrix of type CV_32F. The number of columns is equal to the number of histogram bins (m).
 */
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



