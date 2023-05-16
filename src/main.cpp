
#include "tests.hpp"
#include "imageMatcher.hpp"

int main(int argc, char** argv) {

    // tests processImage uses a flipped image processImage2 identical images
    // processImage3 uses a shifted image
    //processImage2("lenna.jpg");


    //create imagematcher
    ImageMatcher imageMatcher;
    cv::waitKey(0);
    cv::Mat image = cv::imread("../rin.jpg", cv::IMREAD_GRAYSCALE);
    cv::imshow("Query", image);
    std::vector<cv::Mat> images = imageMatcher.siftMatch(image);
    std::cout << "Number of images above threshold: " << images.size() << std::endl;
    for (int i = 0; i < images.size(); i++) {
        cv::imshow("Image", images[i]);
        cv::waitKey(0);
    }
    return 0;
}
