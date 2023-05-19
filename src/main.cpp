
#include "tests.hpp"
#include "imageMatcher.hpp"

int main(int argc, char** argv) {

    // tests processImage uses a flipped image processImage2 identical images
    // processImage3 uses a shifted image
    //processImage2("lenna.jpg");


    //create imagematcher
    ImageMatcher imageMatcher;
    cv::Mat image = cv::imread("../rin.jpg", cv::IMREAD_GRAYSCALE);
    cv::imshow("Query", image);

    //Search images using sift
//    std::cout << "Perform SIFT matching" << std::endl;
//    std::vector<cv::Mat> siftImages = imageMatcher.siftMatch(image);
//    std::cout << "Number of images above threshold: " << siftImages.size() << std::endl;
//    for (int i = 0; i < siftImages.size(); i++) {
//        cv::imshow("Image", siftImages[i]);
//        cv::waitKey(0);
//    }
    //Search images using sGlOH2
    std::cout << "Perform sGLOH2 matching" << std::endl;
    std::vector<cv::Mat> sgloh2Images = imageMatcher.sGLOHMatch(image, M);
    std::cout << "Number of images above threshold: " << sgloh2Images.size() << std::endl;
    for (int i = 0; i < sgloh2Images.size(); i++) {
        cv::imshow("Image", sgloh2Images[i]);
        cv::waitKey(0);
    }
    return 0;
}
