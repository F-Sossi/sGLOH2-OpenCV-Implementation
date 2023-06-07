/**
 * @file sGLOH2.hpp
 * @author Frank Sossi, Justin Boyer
 * @date 5/10/23
 * @brief Definition of the sGLOH2 class.
 *
 * This class provides the definition for the sGLOH2 class, including
 * constructors, computation and utility methods. The class is designed
 * for computing the sGLOH2 descriptor for a given image patch.
 *
 * Implemented functions:
 * sGLOH2::sGLOH2(int m = 4)
 * void sGLOH2::compute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
 * cv::Mat sGLOH2::compute_sGLOH(const cv::Mat& patch)
 * double sGLOH2::distance(const cv::Mat& H_star1, const cv::Mat& H_star2)
 * cv::Mat sGLOH2::cyclicShift(const cv::Mat &descriptor, int k)
 * cv::Mat sGLOH2::compute_sGLOH_single(const cv::Mat &patch)
 * cv::Mat sGLOH2::computeHistogram(const cv::Mat &patch, const cv::Mat &mask, int m)
 *
 * This work is based on the following paper:
 * Bellavia, Fabio, and Carlo Colombo. "Rethinking the sGLOH descriptor." IEEE Transactions on Pattern
 * Analysis and Machine Intelligence 40.4 (2017): 931-944.
 *
 * Bellavia, Fabio, Domenico Tegolo, and Emanuele Trucco. "Improving SIFT-based descriptors stability
 * to rotations." 2010 20th International Conference on Pattern Recognition. IEEE, 2010.
 *
 */
#ifndef SGLOH_OPENCV_SGLOH2_HPP
#define SGLOH_OPENCV_SGLOH2_HPP

#include <opencv2/core.hpp>

/**
 * @class sGLOH2
 * @brief Class for computing the sGLOH2 descriptor for a given image patch.
 */
class sGLOH2 {
private:
    //int m;  ///< discretization of the rotation
    std::vector<std::vector<cv::Mat>> region_masks;

    /**
     * @brief Shifts a descriptor in a cyclic manner.
     * @param descriptor The descriptor to be shifted.
     * @param k The number of positions to shift.
     * @return The cyclically shifted descriptor.
     */
    static cv::Mat cyclicShift(const cv::Mat &descriptor, int k);

    /**
     * @brief Computes the sGLOH descriptor for a single patch.
     * @param patch The patch to compute the descriptor for.
     * @return The computed sGLOH descriptor.
     */
    cv::Mat compute_sGLOH_single(const cv::Mat &patch);

    /**
     * @brief Computes the histogram of the given patch.
     * @param patch The patch to compute the histogram for.
     * @param mask The mask to apply to the patch.
     * @param m The number of bins in the histogram.
     * @return The computed histogram.
     */
    cv::Mat computeHistogram(const cv::Mat &patch, const cv::Mat &mask);

    /**
     * @brief Computes the sGLOH descriptor for the given patch.
     * @param patch The patch to compute the descriptor for.
     * @return The computed sGLOH descriptor.
     */
    cv::Mat compute_sGLOH(const cv::Mat& patch);

public:

    /**
     * @brief Constructs an sGLOH2 object.
     * @param m The number of bins. Default is 4.
     */
    explicit sGLOH2();

    /**
     * @brief Computes the sGLOH2 descriptors for the keypoints in the given image.
     * @param image The image to compute the descriptors for.
     * @param keypoints The keypoints in the image.
     * @param descriptors The output sGLOH2 descriptors.
     */
    void compute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

    /**
     * @brief Computes the distance between two descriptors.
     * @param H_star1 The first descriptor.
     * @param H_star2 The second descriptor.
     * @return The distance between H_star1 and H_star2.
     */
    static double distance(const cv::Mat& H_star1, const cv::Mat& H_star2);

};

#endif //SGLOH_OPENCV_SGLOH2_HPP
/**
 * This work is based on the following paper:
 * Bellavia, Fabio, and Carlo Colombo. "Rethinking the sGLOH descriptor." IEEE Transactions on Pattern
 * Analysis and Machine Intelligence 40.4 (2017): 931-944.
 *
 * Bellavia, Fabio, Domenico Tegolo, and Emanuele Trucco. "Improving SIFT-based descriptors stability
 * to rotations." 2010 20th International Conference on Pattern Recognition. IEEE, 2010.
 *
 */

