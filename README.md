# UVface++

## Description
UVface++ is a real-time face recognition software that operates efficiently without requiring complex hardware, such as GPUs. It performs robustly under challenging conditions, including variations in lighting, pose, and expression. The source code is entirely written in C++ and leverages OpenMP for parallelization.

<div align="center">
    <img src="docs/UVface.gif" alt="UVface GIF">
</div>

## Installation and Execution (Linux)

To run the algorithm, first clone the repository:

```bash
git clone https://github.com/roggerfq/UVface.git
```

Navigate to the `UVface` folder and build the Docker image with the following command:

```bash
docker build -t uvface .
```

Finally, in the next command replace HOST_PATH with a path you want to share with the container and execute to launch UVface++:

```bash
sudo docker run --privileged -e DISPLAY \
    --env="QT_X11_NO_MITSHM=1" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v HOST_PATH:/home/host \
    --device=/dev/video0:/dev/video0 \
    -it uvface
```

## Installation and Execution (Windows)

If the container is executed inside a Linux system running on WSL2, it will not have access to USB cameras. In this case, it is necessary to run the Python script [`stream.py`](/stream/stream.py) on windows to capture video from the camera and send it to `http://172.20.16.1:5000/video_feed`:

```cmd
python stream.py
```

Finally, in the next command replace HOST_PATH with a path you want to share with the container and execute to launch UVface++:

```bash
sudo docker run --privileged -e DISPLAY \
    --env="QT_X11_NO_MITSHM=1" \
    -e DEVICE_URL="http://172.20.16.1:5000/video_feed" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v HOST_PATH:/home/host \
    --device=/dev/video0:/dev/video0 \
    -it uvface
```

**⚠️ Note:** To read the video from the URL, the 'device' field in the GUI must be set to `-1`.

### Face Detection

The face detection process is a cascade of classifiers (see [XML file](cascading_classifiers/clasificador_9_12102_unconstrained_f_max_0_2_evaluation.xml)) constructed using [UVtrainer](https://github.com/roggerfq/UVtrainer). The cascade is evaluated at multiple scales across the image. Each stage of the cascade consists of an ensemble of regression tree classifiers that use NPD features for evaluation [1]. The following diagram provides an overview of the face detection algorithm:

<div align="center">
    <img src="docs/face_detection.svg" alt="face_detection">
</div>

The video below demonstrates how to load and configure the cascade classifier for face detection:

[Insert Video Here]

### Face Recognition

The face recognition process in UVface++ implements the algorithm published in [2], which comprises two stages: feature extraction and classification.

1. **Feature Extraction:**
   - This stage employs the Affine Covariant Region Detector [3] to identify key points on the face image. These key points are invariant to various geometric and photometric transformations.
   - Around each key point, a Gabor feature vector of 1296 dimensions is computed. Dimensionality reduction is then performed using PCA, resulting in a 128-dimensional feature vector.

2. **Classification:**
   - The extracted features from the test image are compared against those in the dataset using Sparse Representation Classification (SRC) [4].

The following diagram provides an overview of the face recognition process:

<div align="center">
    <img src="docs/face_recognition.svg" alt="face_recognition">
</div>

The video below demonstrates how to build a simple dataset and use the software for face recognition:

[Insert Video Here]

## Interface Guide

The next figure shows the main interface along with the most important panels: the face detector configuration and the face recognizer configuration GUIs.

<div align="center">
    <img src="docs/main_interface.svg" alt="face_detection">
</div>

### Face detector configuration

This interface allows loading a cascade classifier stored in an XML file (this XML file is constructed with UVtrainer). The interface also allows configuring the **Base size** of the search windows, the **Step Factor** with which the search windows move, and the **Scale factor** by which the search window will increase its size after finishing scanning the image. **Maximum size** refers to the maximum allowed dimension (width or height) for an input image. To understand **GroupThreshold** and **EPS**, please refer to the [OpenCV documentation](https://docs.opencv.org/4.x/de/de1/group__objdetect__common.html).

An important configuration is the **Detection degrees**; by default, it is set to 0, but it is possible to detect faces at different angles by entering each angle separated by commas, for example: 30, 15, 0, -15, and -30. Additionally, it is possible to enable **Normalize the rotations**, which means that the detections will adjust the best angle for the faces detected. 

The interface also allows setting the checkbox for **Activate skin color**, which activates a simple skin color algorithm and may help improve computation speed and reduce the false positive rate.

### Face recognizer configuration

This panel allows loading or creating a database. In both cases, the first step is clicking the **Load dataset** button. If an empty folder or a folder with subfolders containing images is selected, the software will configure this folder as a database. If a folder that has been previously configured as a database is selected, the software will load the data. After that, the panel allows adding or deleting people, and adding images to each user either from files or by capturing images from a camera (see Setting the camera). The panel also allows cropping images using the mouse in the video panel and resizing individual images, all images of a user, or all images in the dataset.

This panel includes three additional tabs:

#### Description Tab

<div align="center">
    <img src="docs/descriptor_config_gui.svg" alt="descriptor_config_gui">
</div>

The **Descriptor** tab allows selecting the descriptor to use. If **GTP (Gabor Ternary Pattern)** is selected, it is possible to compute the PCA matrix with features extracted from the current dataset or use a default PCA matrix computed with 1,000 individuals randomly selected from the [LFW dataset](https://vis-www.cs.umass.edu/lfw/).

#### SRC Tab

<div align="center">
    <img src="docs/SRC_config_gui.svg" alt="SRC_config_gui">
</div>

The **SRC** tab allows configuring the parameters of the sparse solution. Key parameters include:

- **Fast filtering**: The maximum number of descriptors extracted from a test image to compute the sparse solution. The descriptors selected are those with the highest correlation with the descriptors in the database.
- **Recognition threshold**: This is compared with the confidence score given by the system to a test image. If the confidence score is higher than the threshold, the system will recognize the test image as one of the individuals in the dataset;  otherwise, it will identify the test image as "Unknown".

For more details about the other parameters in this tab, consult reference [4].

#### Test Tab

<div align="center">
    <img src="docs/test_config_gui.svg" alt="descriptor_config_gui">
</div>

The **Test** tab allows configuring the size to which the images will be rescaled before being processed by the face recognition system. A larger size improves accuracy but reduces computational performance.

This tab also provides the option to test only the face recognition system with images acquired from files or captured from a camera. When an image is tested, the **Result per Image** panel will display various metrics associated with the face recognition algorithm.

##### Interactive Sparse Solution Plot
The **Plot Sparse Solution** button opens an interactive GUI that allows:
- Viewing the sparse approximation of the descriptors for the tested image.
- Analyzing the components of the sparse solution associated with each individual in the database.
- Observing the reconstruction error for each class in the database.

The following video briefly showcases this GUI:

<p align="center">
  <a href="https://youtu.be/m9HjeDe2bR4">
    <img src="docs/Sparse_solution_plot.png" alt="Custom Demo Video">
  </a>
</p>

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


