
#include "tests.hpp"
#include "imageMatcher.hpp"

int main(int argc, char** argv) {

    // processImage uses a flipped image
    // processImage2 identical images
    // processImage3 uses a shifted image
    // processImage4 uses a rotated image with FLANN matching
    // processImage5 uses a scaled with BF matching
    // processImage6 uses a 45 degree rotation with BF matching
    processImage4("ciri.jpg");


//    //create imagematcher
//    ImageMatcher imageMatcher;
//    cv::Mat image = cv::imread("../rin.jpg", cv::IMREAD_GRAYSCALE);
//    cv::imshow("Query", image);
//
//    //Search images using sift
//    std::cout << "Perform SIFT matching" << std::endl;
//    std::vector<cv::Mat> siftImages = imageMatcher.siftMatch(image);
//    std::cout << "Number of images above threshold: " << siftImages.size() << std::endl;
//    for (int i = 0; i < siftImages.size(); i++) {
//        cv::imshow("Image", siftImages[i]);
//        cv::waitKey(0);
//    }
    //Search images using sGlOH2
//    std::cout << "Perform sGLOH2 matching" << std::endl;
//    std::vector<cv::Mat> sgloh2Images = imageMatcher.sGLOHMatch(image, M);
//    std::cout << "Number of images above threshold: " << sgloh2Images.size() << std::endl;
//    for (int i = 0; i < sgloh2Images.size(); i++) {
//        cv::imshow("Image", sgloh2Images[i]);
//        cv::waitKey(0);
//    }
    return 0;
}
