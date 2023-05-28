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
#ifndef SGLOH_OPENCV_SGLOH2_HPP
#define SGLOH_OPENCV_SGLOH2_HPP

#include <opencv2/core.hpp>

/**
 * @class sGLOH2
 * @brief Class for computing the sGLOH2 descriptor for a given image patch.
 */
class sGLOH2 {
private:
    int m;  ///< discretization of the rotation
    cv::Mat H1, H2;
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
    cv::Mat computeHistogram(const cv::Mat &patch, const cv::Mat &mask, int m);

    /**
     * @brief Computes the cosine similarity between two matrices.
     * @param H1 The first matrix.
     * @param H2 The second matrix.
     * @return The cosine similarity between H1 and H2.
     */
    double cosine_similarity(const cv::Mat &H1, const cv::Mat &H2);

    /**
     * @brief Computes the sGLOH descriptor for the given patch.
     * @param patch The patch to compute the descriptor for.
     * @return The computed sGLOH descriptor.
     */
    cv::Mat compute_sGLOH(const cv::Mat& patch);

public:

    /**
     * @brief Constructs an sGLOH2 object.
     * @param m The number of bins. Default is 8.
     */
    explicit sGLOH2(int m = 8);

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

    /**
     * @brief Computes a custom histogram of the given data.
     * @param data The data to compute the histogram for.
     * @param binEdges The edges of the bins in the histogram.
     * @return The computed histogram.
     */
    cv::Mat computeCustomHistogram(const cv::Mat &data, const std::vector<float> &binEdges);
};

#endif //SGLOH_OPENCV_SGLOH2_HPP

