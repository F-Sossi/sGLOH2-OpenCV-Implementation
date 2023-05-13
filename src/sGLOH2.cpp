//
// Created by user on 5/10/23.
//

#include "sGLOH2.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

constexpr int PATCH_SIZE = 32;

sGLOH2::sGLOH2(int m) : m(m) {
    // Initialize any other member variables as needed
}

void sGLOH2::compute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    // Create a SIFT detector.
    // SIFT (Scale-Invariant Feature Transform) is an algorithm in computer vision to detect and describe local features in images.
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();

    // Detect keypoints using SIFT.
    // Keypoints are points of interest in the image. SIFT finds these points and computes a descriptor for each one.
    detector->detect(image, keypoints);

    // Define the size of the patch to extract around each keypoint.
    // A patch is a small region of the image around a keypoint. The size of the patch is defined by the variable patchSize.
    int patchSize = PATCH_SIZE;  // Adjust this value according to your requirements

    // Compute descriptors for each keypoint.
    // A descriptor is a vector that describes the local features of a keypoint.
    for (auto& keypoint : keypoints) {
        // Calculate the top-left corner of the patch.
        // The patch is a square region of the image centered at the keypoint.
        cv::Point2f topLeft(keypoint.pt.x - patchSize / 2, keypoint.pt.y - patchSize / 2);

        // Check if the patch is within the image boundaries.
        // If the patch is not fully within the image, we skip this keypoint.
        if (topLeft.x < 0 || topLeft.y < 0 || topLeft.x + patchSize > image.cols || topLeft.y + patchSize > image.rows) {
            continue;  // Skip this keypoint if the patch is not fully within the image
        }

        // Extract the patch around the keypoint.
        // The patch is a square region of the image with side length patchSize and top-left corner at topLeft.
        cv::Mat patch = image(cv::Rect(topLeft.x, topLeft.y, patchSize, patchSize));

        // Compute sGLOH descriptor for the patch.
        // The sGLOH descriptor is a modification of the GLOH descriptor that is more robust to changes in rotation.
        cv::Mat descriptor = compute_sGLOH(patch);

        // Append descriptor to descriptors matrix.
        // The descriptors matrix is a matrix where each row is a descriptor of a keypoint.
        descriptors.push_back(descriptor);
    }
}

cv::Mat sGLOH2::compute_sGLOH_single(const cv::Mat& patch) {
    // Initialize the descriptor as a single-row matrix of zeros.
    cv::Mat descriptor = cv::Mat::zeros(1, 0, CV_32F);

    // Compute the radius of the circles.
    int radius = patch.rows / (2 * m);  // The radius is half the patch size divided by m.

    // Compute the histogram for each circular region.
    for (int i = 0; i < m; i++) {
        // Compute the inner and outer radius of the region.
        int inner_radius = i * radius;
        int outer_radius = (i + 1) * radius;

        // Initialize a mask for the region.
        cv::Mat region_mask = cv::Mat::zeros(patch.size(), CV_8U);

        // Set the pixels within the region to 1.
        for (int y = 0; y < patch.rows; y++) {
            for (int x = 0; x < patch.cols; x++) {
                int dx = x - patch.cols / 2;
                int dy = y - patch.rows / 2;
                int distance = std::sqrt(dx * dx + dy * dy);
                if (distance >= inner_radius && distance < outer_radius) {
                    region_mask.at<uchar>(y, x) = 1;
                }
            }
        }

        // Compute the histogram for the region.
        cv::Mat histogramMat = computeHistogram(patch, region_mask, m);

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

//double sGLOH2::distance(const cv::Mat& H_star1, const cv::Mat& H_star2) {
//    // Initialize the minimum distance to the maximum possible value.
//    double min_distance = std::numeric_limits<double>::max();
//
//    // Separate the two descriptors in H_star1 and H_star2 into two halves each.
//    // H1_1 and H1_2 are the two halves of the first descriptor.
//    // H2_1 and H2_2 are the two halves of the second descriptor.
//    cv::Mat H1_1 = H_star1(cv::Range(0, 1), cv::Range(0, H_star1.cols/2));
//    cv::Mat H1_2 = H_star1(cv::Range(0, 1), cv::Range(H_star1.cols/2, H_star1.cols));
//    cv::Mat H2_1 = H_star2(cv::Range(0, 1), cv::Range(0, H_star2.cols/2));
//    cv::Mat H2_2 = H_star2(cv::Range(0, 1), cv::Range(H_star2.cols/2, H_star2.cols));
//
//    // Compute the cyclic shifts of H2_1 and H2_2.
//    // This is done to account for the rotation invariance of the descriptor.
//    for(int k = 0; k < m; ++k) {
//        // Shift H2_1 and H2_2 by k positions.
//        cv::Mat H2_1_shifted = cyclicShift(H2_1, k);
//        cv::Mat H2_2_shifted = cyclicShift(H2_2, k);
//
//        // Concatenate the shifted descriptors to form the complete descriptor.
//        cv::Mat H2_shifted;
//        cv::hconcat(H2_1_shifted, H2_2_shifted, H2_shifted);
//
//        // Compute the distance between H_star1 and the shifted H2.
//        // The L2 norm (Euclidean distance) is used as the distance metric.
//        double distance = cv::norm(H_star1, H2_shifted, cv::NORM_L2);
//
//        // Update the minimum distance if the current distance is smaller.
//        if (distance < min_distance) {
//            min_distance = distance;
//        }
//    }
//
//    // Return the minimum distance found.
//    // This is the distance between H_star1 and the best match among the cyclic shifts of H_star2.
//    return min_distance;
//}
double sGLOH2::distance(const cv::Mat& H_star1, const cv::Mat& H_star2) {
    // Initialize the minimum distance to the maximum possible value.
    double min_distance = std::numeric_limits<double>::max();
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

        // Compute the distance between H_star1 and the shifted H2.
        double distance = cv::norm(H_star1, H2_shifted, cv::NORM_L2);

        // Update the minimum distance and best rotation if the current distance is smaller.
        if (distance < min_distance) {
            min_distance = distance;
            best_rotation = k;
        }
    }

    // Reject incongruous correspondences by adding a global constraint on the rotations.
    int rotation_threshold = 5;  // Adjust this value according to your requirements
    if (abs(best_rotation - m/2) > rotation_threshold) {
        return std::numeric_limits<double>::max();
    }

    // Return the minimum distance found.
    return min_distance;
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

//cv::Mat sGLOH2::computeHistogram(const cv::Mat& region, int m) {
//    // Compute the gradient of the region. The Sobel function is used to find the intensity
//    // gradient of the image. For each pixel, a 2D spatial gradient measurement is computed
//    // for an image, which gives the orientation and magnitude of the gradient at each point.
//    cv::Mat grad_x, grad_y;
//    cv::Sobel(region, grad_x, CV_32F, 1, 0, 3);
//    cv::Sobel(region, grad_y, CV_32F, 0, 1, 3);
//
//    // Compute the magnitude and orientation of the gradient. The cartToPolar function
//    // calculates the magnitude and angle of 2D vectors. The angle is computed in radians
//    // and the magnitude is computed as sqrt(x^2 + y^2).
//    cv::Mat magnitude, orientation;
//    cv::cartToPolar(grad_x, grad_y, magnitude, orientation, true);
//
//    // Convert orientation from radians to degrees. This is done because the histogram
//    // function expects input in degrees, not radians.
//    orientation = orientation * 180.0 / CV_PI;
//
//    // Compute the histogram of orientations. The calcHist function calculates the
//    // histogram of a set of arrays. It can operate on arrays of arbitrary dimensions
//    // and depth. In this case, it's used to compute the histogram of gradient orientations.
//    cv::Mat histogram;
//    int histSize = m;  // Number of bins
//    float range[] = { 0, 360 } ;  // Orientation ranges from 0 to 360 degrees
//    const float* histRange = { range };
//    cv::calcHist(&orientation, 1, 0, cv::Mat(), histogram, 1, &histSize, &histRange);
//
//    // Reshape the histogram to a single-row matrix. This is done to ensure that the
//    // histogram can be easily concatenated with other histograms or used in further
//    // computations as a single-row matrix.
//    histogram = histogram.reshape(1, 1);
//
//    return histogram;
//}

//cv::Mat sGLOH2::compute_sGLOH_single(const cv::Mat& patch) {
//    // Initialize the descriptor as a single-row matrix of zeros.
//    // The type of the matrix is CV_32F, which means that it contains 32-bit floating-point numbers.
//    cv::Mat descriptor = cv::Mat::zeros(1, 0, CV_32F);
//
//    // Compute the size of the regions into which the patch will be divided.
//    // Each region will be a square with side length equal to the number of rows in the patch divided by m.
//    int region_size = patch.rows / m;
//
//    // Compute the histogram for each region.
//    // The patch is divided into m*m regions, and a histogram is computed for each one.
//    for (int r = 0; r < m; r++) {
//        for (int d = 0; d < m; d++) {
//            // Compute the region.
//            // The region is a square with top-left corner at (r*region_size, d*region_size) and side length region_size.
//            cv::Rect region(r * region_size, d * region_size, region_size, region_size);
//
//            // Compute the histogram for the region.
//            // The computeHistogram function computes a histogram of gradient orientations for the region.
//            cv::Mat histogramMat = computeHistogram(patch(region), m);
//
//            // Concatenate the histogram to the descriptor.
//            // The hconcat function horizontally concatenates the histogram to the descriptor.
//            // This means that the histogram is added as new columns at the end of the descriptor.
//            cv::hconcat(descriptor, histogramMat, descriptor);
//        }
//    }
//
//    // Return the final descriptor.
//    // The descriptor is a single-row matrix that contains the concatenated histograms of all regions in the patch.
//    return descriptor;
//}

// Uses custom histogram function
//cv::Mat sGLOH2::compute_sGLOH_single(const cv::Mat& patch) {
//    cv::Mat descriptor = cv::Mat::zeros(1, 0, CV_32F);
//
//    // Compute the size of the regions
//    int region_size = patch.rows / m;
//
//    // Define the bin edges for the histogram
//    std::vector<float> binEdges = {0, 90, 180, 270, 360};  // Fill this with the desired bin edges
//
//    // Compute the histogram for each region
//    for (int r = 0; r < m; r++) {
//        for (int d = 0; d < m; d++) {
//            // Compute the region
//            cv::Rect region(r * region_size, d * region_size, region_size, region_size);
//
//            // Compute the histogram for the region
//            cv::Mat histogramMat = computeCustomHistogram(patch(region), binEdges);
//
//            // Concatenate the histogram to the descriptor.
//            cv::hconcat(descriptor, histogramMat, descriptor);
//        }
//    }
//
//    return descriptor;
//}

//// Compute using grid
//void sGLOH2::compute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
//    // Define grid size
//    int gridSize = 32;  // adjust this value according to your requirements
//
//    // Generate keypoints in a grid
//    keypoints.clear();
//    for (int y = gridSize / 2; y < image.rows; y += gridSize) {
//        for (int x = gridSize / 2; x < image.cols; x += gridSize) {
//            keypoints.push_back(cv::KeyPoint(cv::Point2f(x, y), gridSize));
//        }
//    }
//
//    // Define the size of the patch to extract around each keypoint
//    int patchSize = gridSize;  // Patch size should match grid size
//
//    // Compute descriptors for each keypoint
//    for (auto& keypoint : keypoints) {
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

//cv::Mat sGLOH2::compute_sGLOH(const cv::Mat& patch) {
//    // Compute the first sGLOH descriptor on the original patch
//    cv::Mat descriptor1 = compute_sGLOH_single(patch);
//
//    // Rotate the patch by pi/m degrees
//    cv::Mat patch_rotated;
//    cv::Point2f center(patch.cols/2.0, patch.rows/2.0);
//    double angle = 180.0 / m;
//    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
//    // try cv::INTER_NEAREST and cv::INTER_CUBIC default is cv::INTER_LINEAR
//    cv::warpAffine(patch, patch_rotated, rotation_matrix, patch.size(), cv::INTER_NEAREST);
//
//    // Compute the second sGLOH descriptor on the rotated patch
//    cv::Mat descriptor2 = compute_sGLOH_single(patch_rotated);
//
//    // Concatenate the two descriptors
//    cv::Mat descriptor;
//    cv::hconcat(descriptor1, descriptor2, descriptor);
//
//    // Normalize the descriptor
//    cv::normalize(descriptor, descriptor, 1, 0, cv::NORM_L1);
//
//    return descriptor;
//}

//double sGLOH2::distance(const cv::Mat& H_star1, const cv::Mat& H_star2) {
//    double min_distance = std::numeric_limits<double>::max();
//
//    // Separate the two descriptors in H_star1 and H_star2
//    // H1_1 and H1_2 are the two halves of the first descriptor
//    // H2_1 and H2_2 are the two halves of the second descriptor
//    cv::Mat H1_1 = H_star1(cv::Range(0, 1), cv::Range(0, H_star1.cols/2));
//    cv::Mat H1_2 = H_star1(cv::Range(0, 1), cv::Range(H_star1.cols/2, H_star1.cols));
//    cv::Mat H2_1 = H_star2(cv::Range(0, 1), cv::Range(0, H_star2.cols/2));
//    cv::Mat H2_2 = H_star2(cv::Range(0, 1), cv::Range(H_star2.cols/2, H_star2.cols));
//
//    // Compute the cyclic shifts of H2_1 and H2_2
//    for(int k = 0; k < m; ++k) {
//        // Shift H2_1 and H2_2
//        cv::Mat H2_1_shifted = cyclicShift(H2_1, k);
//        cv::Mat H2_2_shifted = cyclicShift(H2_2, k);
//
//        // Concatenate the shifted descriptors
//        cv::Mat H2_shifted;
//        cv::hconcat(H2_1_shifted, H2_2_shifted, H2_shifted);
//
//        // Compute the distance to H_star1
//        double distance = cv::norm(H_star1, H2_shifted, cv::NORM_L2);
//
//        // Update the minimum distance
//        if (distance < min_distance) {
//            min_distance = distance;
//        }
//    }
//
//    return min_distance;
//}

//cv::Mat sGLOH2::cyclicShift(const cv::Mat& descriptor, int k) {
//    cv::Mat shifted_descriptor = descriptor.clone();
//
//    // Compute the size of the blocks
//    int block_size = descriptor.cols / m;
//
//    // Shift each block by k positions
//    for (int i = 0; i < m; ++i) {
//        cv::Mat block = descriptor(cv::Range(0, 1), cv::Range(i * block_size, (i + 1) * block_size));
//
//        // Convert the block to a vector
//        std::vector<float> block_vec;
//        block.copyTo(block_vec);
//
//        // Rotate the block vector by k positions
//        std::rotate(block_vec.begin(), block_vec.begin() + k, block_vec.end());
//
//        // Convert the vector back to a Mat and replace the block in the descriptor
//        cv::Mat shifted_block = cv::Mat(block_vec).reshape(1, 1);
//        shifted_block.copyTo(shifted_descriptor(cv::Range(0, 1), cv::Range(i * block_size, (i + 1) * block_size)));
//    }
//
//    return shifted_descriptor;
//}

//cv::Mat sGLOH2::computeCustomHistogram(const cv::Mat& data, const std::vector<float>& binEdges) {
//    std::vector<int> histogram(binEdges.size() - 1, 0);
//
//    for (int i = 0; i < data.rows; ++i) {
//        for (int j = 0; j < data.cols; ++j) {
//            float value = data.at<float>(i, j);
//
//            // Find the bin that this value belongs to
//            for (size_t bin = 0; bin < binEdges.size() - 1; ++bin) {
//                if (value >= binEdges[bin] && value < binEdges[bin + 1]) {
//                    ++histogram[bin];
//                    break;
//                }
//            }
//        }
//    }
//
//    // Convert the histogram to a cv::Mat and ensure it's a single-row matrix.
//    cv::Mat histogramMat = cv::Mat(histogram).reshape(1, 1);
//
//    // Ensure that the histogramMat has the same data type as the descriptor matrix
//    histogramMat.convertTo(histogramMat, CV_32F);
//
//    return histogramMat;
//}

//// Use radians to compute the histogram
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
//    // Convert orientation from radians to degrees
//    orientation = orientation * 180.0 / CV_PI;
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

//cv::Mat sGLOH2::compute_sGLOH_single(const cv::Mat& patch) {
//    cv::Mat descriptor = cv::Mat::zeros(1, 0, CV_32F);
//
//    // Compute the size of the regions
//    int region_size = patch.rows / m;
//
//    // Define the bin edges for the histogram
//    std::vector<float> binEdges = {0, 90, 180, 270, 360};  // Fill this with the desired bin edges
//
//    // Compute the histogram for each region
//    for (int r = 0; r < m; r++) {
//        for (int d = 0; d < m; d++) {
//            // Compute the region
//            cv::Rect region(r * region_size, d * region_size, region_size, region_size);
//
//            // Compute the histogram for the region
//            cv::Mat histogramMat = computeCustomHistogram(patch(region), binEdges);
//
//            // Concatenate the histogram to the descriptor.
//            cv::hconcat(descriptor, histogramMat, descriptor);
//        }
//    }
//
//    return descriptor;
//}
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
