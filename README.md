# UVface++

## Description
UVface++ is a software solution designed for real-time human face detection and recognition without the need for complex hardware, such as GPUs. It performs robustly under challenging conditions, including variations in lighting, occlusion, pose, and expression. Remarkably, the system can operate effectively with only one training image per person. The source code is entirely developed in C++ and leverages OpenMP for parallelization.

## Installation and Execution

To run the algorithm, first clone the repository:

```bash
git clone https://github.com/roggerfq/UVface.git
```

Navigate to the `UVface` folder and build the Docker image with the following command:

```bash
docker build -t uvface .
```

Finally, execute the following command to launch UVface++:

```bash
sudo docker run --privileged -e DISPLAY \
    --env="QT_X11_NO_MITSHM=1" \
    -e DEVICE_URL="http://172.20.16.1:5000/video_feed" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --device=/dev/video0:/dev/video0 \
    -it uvface
```

This command grants the container necessary privileges, activates the QT graphical interface options, and configures a video URL for UVface++ to consume video feeds. This setup is essential for Windows systems where USB camera devices cannot be shared directly with Linux running in WSL2. For systems running Docker directly on Linux, the command also shares the USB camera device with Docker.

The image below illustrates the main interface of UVface++ running on Ubuntu 18.04 through WSL2:

### Face Detection

The face detection stage employs a cascade of classifiers constructed using UVtrainer, implementing the algorithm described in [1]. The cascade is evaluated at multiple scales across the image. Each stage of the cascade consists of an ensemble of regression tree classifiers that use NPD features for evaluation. The following diagram provides an overview of the face detection process:

[Insert Diagram Here]

The video below demonstrates how to load and configure the cascade classifier for face detection:

[Insert Video Here]

### Face Recognition

Face recognition in UVface++ comprises two stages: feature extraction and classification.

1. **Feature Extraction:**
   - This stage employs the Affine Covariant Region Detector to identify key points on the face image. These key points are invariant to various geometric and photometric transformations.
   - Around each key point, a Gabor feature vector of 1296 dimensions is computed. Dimensionality reduction is then performed using PCA, resulting in a 128-dimensional feature vector.

2. **Classification:**
   - The extracted features from the test image are compared against those in the dataset using Sparse Representation Classification (SRC).

The following diagram provides an overview of the face recognition process:

[Insert Diagram Here]

The video below demonstrates how to build a simple dataset and use the software for face recognition from different sources, such as images, cameras, and videos:

[Insert Video Here]

## Extending the System

You can integrate a new descriptor into the system using the following template:

```cpp
#ifndef MY_NEW_DESCRIPTOR_H
#define MY_NEW_DESCRIPTOR_H

#include "descriptor.h" // Abstract base class
#include <opencv2/opencv.hpp> // Include OpenCV

class MyNewDescriptor : public ABSTRACT_DESCRIPTOR
{
    // Add private members here
public:
    MyNewDescriptor(); // Constructor
    ~MyNewDescriptor(); // Destructor

    // Virtual functions to override
    cv::Mat* baseDescriptor(const cv::Mat &img) override;
    void postProcessing(cv::Mat &baseDescriptor, cv::Mat &finalDescriptor, std::vector<int> &ithRows) override;
    Eigen::MatrixXf* test(const cv::Mat &img) override;
};

#endif
```

### Implementation Example

```cpp
#include "MyNewDescriptor.h"

MyNewDescriptor::MyNewDescriptor() {
    // Constructor implementation
}

MyNewDescriptor::~MyNewDescriptor() {
    // Destructor implementation
}

cv::Mat* MyNewDescriptor::baseDescriptor(const cv::Mat &img) {
    // Implement descriptor extraction
}

void MyNewDescriptor::postProcessing(cv::Mat &baseDescriptor, cv::Mat &finalDescriptor, std::vector<int> &ithRows) {
    // Implement post-processing
}

Eigen::MatrixXf* MyNewDescriptor::test(const cv::Mat &img) {
    cv::Mat *aux = baseDescriptor(img); // Extract base descriptors
    // Implement testing logic
    return aux; // Ensure proper memory management
}
```

Finally, include the new descriptor in the recognizer:

```cpp
#include "MyNewDescriptor.h"

RECOGNIZER_FACIAL::RECOGNIZER_FACIAL() {
    MyNewDescriptor *newDescriptor = new MyNewDescriptor;
    descriptors.push_back(newDescriptor);
}
```

## Author
Roger Figueroa Quintero - [LinkedIn Profile](https://www.linkedin.com/in/roger-figueroa-quintero/)

## License
This project is licensed under the [MIT License](LICENSE.md), allowing unrestricted use, modification, and distribution under the terms of the license.

## References

[1] S. Liao, A. K. Jain, and S. Z. Li, "A fast and accurate unconstrained face detector," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 38, no. 2, pp. 211-223, 2015.

[2] S. Liao, A. K. Jain, and S. Z. Li, "Partial face recognition: Alignment-free approach," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 35, no. 5, pp. 1193-1205, 2012.

[3] K. Mikolajczyk and C. Schmid, "Scale and affine invariant interest point detectors," *Int. J. Comput. Vision*, vol. 60, no. 1, pp. 63-86, 2004. [Online]. Available: https://www.robots.ox.ac.uk/~vgg/research/affine/detectors.html.

[4] Y. Li, "Coordinate descent optimization for l1 minimization with application to compressed sensing; a greedy algorithm solving the unconstrained problem," *Inverse Problems and Imaging*, vol. 3, pp. 1-17, 2009.

