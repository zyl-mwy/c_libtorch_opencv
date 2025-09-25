#include "iostream"
#include "opencv2/opencv.hpp"
#include "torch/script.h"
#include "fstream"

void checkPath(const char* path) {
    std::ifstream in;
    in.open(path);
    bool flag = (bool)in;
    in.close();
    if (flag) return;
    else {
        std::cout << "file " << path << " doesn't exist!" << std::endl;
        exit(-1);
    }
}

int main(int argc, char const *argv[])
{
    if (argc != 3) {
        std::cout << "usage : digit <model path> <image path>" << std::endl;
        return -1;
    }

    checkPath(argv[1]);
    checkPath(argv[2]);
    cv::Mat img = cv::imread(argv[2]), gimg, fimg, rimg;
    cv::cvtColor(img, gimg, cv::COLOR_BGR2GRAY);

    gimg.convertTo(fimg, CV_32F, - 1. / 255., 1.);
    cv::resize(fimg, rimg, {8, 8});

    // convert Mat to tensor
    at::Tensor img_tensor = torch::from_blob(
        rimg.data,
        {1, 1, 8, 8},
        torch::kFloat32
    );

    // load model
    torch::jit::Module model = torch::jit::load(argv[1]);

    // torch.no_grad()
    torch::NoGradGuard no_grad;
    
    // forward
    torch::Tensor out = model({img_tensor}).toTensor();
    int pre_lab = torch::argmax(out, 1).item().toInt();

    std::cout << "predict number is " << pre_lab << std::endl;
    return 0;
}