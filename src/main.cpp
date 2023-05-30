#include "ImageComparatorSgloh.hpp"
#include "ImageComparatorSift.hpp"
#include "sGlohUtilites.hpp"


#include <chrono>

int main(int argc, char** argv) {

//--------------------Single Image Tests-----------------------
    // processImage uses a flipped image
    // processImage2 identical images
    // processImage3 uses a shifted image
    // processImage4 uses a rotated image with FLANN matching
    // processImage5 uses a scaled with BF matching
    // processImage6 uses a 45-degree rotation with BF matching

//-------Two image test ---------------------------------------

//    compareImages("img1.ppm", "img2.ppm");


//-------Image Search tests -----------------------------------

    // Time the execution of the sGLOH2 descriptor
    // begin timer
    auto start = std::chrono::high_resolution_clock::now();
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
    ImageComparatorSgloh comparator(imageInputPath, folderPath);
    comparator.runComparison(skipInputs);

    // end timer
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "sGLOH2 descriptor took " << elapsed.count() << " seconds" << std::endl;

    // Time the execution of the SIFT descriptor
    // begin timer
    auto start2 = std::chrono::high_resolution_clock::now();

    ImageComparatorSift comparatorSift(imageInputPath, folderPath);
    comparatorSift.runComparison(skipInputs);

    // end timer
    auto finish2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = finish2 - start2;
    std::cout << "SIFT descriptor took " << elapsed2.count() << " seconds" << std::endl;
    !skipInputs && cv::waitKey(0);
    return 0;
}
