/**
 * @file main.cpp.cpp
 * @author Frank Sossi, Justin Boyer
 * @date 5/10/23
 * @brief main file for running sGLOH2 descriptor
 * This file provides the main function for running the sGLOH2 descriptor
 * on a given image. The user can provide the path to the image as an
 * argument, or the default image will be used. The user can also provide
 * the path to the folder containing the images to compare against, or the
 * default folder will be used.
 *
 * Implemented functions:
 * int main(int argc, char** argv)
 *
 * This work is based on the following paper:
 * Bellavia, Fabio, and Carlo Colombo. "Rethinking the sGLOH descriptor." IEEE Transactions on Pattern
 * Analysis and Machine Intelligence 40.4 (2017): 931-944.
 *
 * Bellavia, Fabio, Domenico Tegolo, and Emanuele Trucco. "Improving SIFT-based descriptors stability
 * to rotations." 2010 20th International Conference on Pattern Recognition. IEEE, 2010.
 *
 */
#include "ImageComparatorSgloh.hpp"
#include "ImageComparatorSift.hpp"

int main(int argc, char** argv) {

    //Get input image path from arguments if provided
    std::string imageInputPath;
    std::string folderPath;
    bool skipInputs = false; // skip user input for run if starting from script
    if (argc == 3) {
        imageInputPath = argv[1];
        folderPath = argv[2];
        skipInputs = true;
    } else {
        imageInputPath = "../src_img/toucan.png";
        folderPath = "../images";
    }
    if(skipInputs){
        std::cout << "Testing: " << imageInputPath << "\n" << std::endl;
    }
    // Run sGLOH2 descriptor
    ImageComparatorSgloh comparator(imageInputPath, folderPath);
    comparator.runComparison(skipInputs);

    // Run SIFT descriptor
    ImageComparatorSift comparatorSift(imageInputPath, folderPath);
    comparatorSift.runComparison(skipInputs);

    !skipInputs && cv::waitKey(0);

    return 0;
}
