
#include "tests.hpp"
#include "imageMatcher.hpp"
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

    ImageComparatorSgloh comparator("../src_img/1_r55.png", "../images");
    comparator.runComparison();

    // end timer
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "sGLOH2 descriptor took " << elapsed.count() << " seconds" << std::endl;

    // Time the execution of the SIFT descriptor
    // begin timer
    auto start2 = std::chrono::high_resolution_clock::now();

    ImageComparatorSift comparatorSift("../src_img/1_r55.png", "../images");
    comparatorSift.runComparison();

    // end timer
    auto finish2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = finish2 - start2;
    std::cout << "SIFT descriptor took " << elapsed2.count() << " seconds" << std::endl;


//-------Image Search tests 2-----------------------------------

//    //create imagematcher
//    ImageMatcher imageMatcher;
//    cv::Mat image = cv::imread("../rin.jpg", cv::IMREAD_GRAYSCALE);
//    cv::imshow("Query", image);
//
//    //Search images using sift
////    std::cout << "Perform SIFT matching" << std::endl;
////    std::vector<cv::Mat> siftImages = imageMatcher.siftMatch(image);
////    std::cout << "Number of images above threshold: " << siftImages.size() << std::endl;
////    for (int i = 0; i < siftImages.size(); i++) {
////        cv::imshow("Image", siftImages[i]);
////        cv::waitKey(0);
////    }
//    //Search images using sGlOH2
//    std::cout << "Perform sGLOH2 matching" << std::endl;
//    std::vector<cv::Mat> sgloh2Images = imageMatcher.sGLOHMatch(image, M);
//    std::cout << "Number of images above threshold: " << sgloh2Images.size() << std::endl;
//    for (int i = 0; i < sgloh2Images.size(); i++) {
//        cv::imshow("Image", sgloh2Images[i]);
//        cv::waitKey(0);
//    }
    return 0;
}
