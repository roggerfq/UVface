# Use the base image of Ubuntu 16.04
FROM ubuntu:16.04

# Set non-interactive timezone to avoid questions during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    unzip \
    cmake \
    git \
    libqt4-dev \
    libopencv-dev \
    libgtk2.0-dev \
    pkg-config \
    python-dev \
    python-numpy \
    python-opencv \
    wget

# Download and compile OpenCV 2.4
RUN cd /tmp && \
    wget https://github.com/opencv/opencv/archive/2.4.13.6.zip && \
    unzip 2.4.13.6.zip && \
    cd opencv-2.4.13.6 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install

# Set the necessary environment variables for Qt4
ENV QT_X11_NO_MITSHM 1
ENV DISPLAY :0

# Install Qt4.8
RUN apt-get install -y qt4-default

# Create the UVface_build folder in /home
RUN mkdir -p /home/UVface_build

# Create the UVface folder in /home for development purposes
#RUN mkdir -p /home/UVface

# Copy all files and folders (except the Dockerfile) to the UVface_build directory
COPY . /home/UVface_build/
RUN rm /home/UVface_build/Dockerfile

# Download, extract, and keep the specific file
RUN cd /tmp && \
    wget https://thor.robots.ox.ac.uk/affine/extract_features2.tar.gz && \
    tar -xzvf extract_features2.tar.gz && \
    mv extract_features/extract_features_64bit.ln /home/UVface_build/ && \
    rm -rf /tmp/*

# Set the working directory to UVface_build/build
WORKDIR /home/UVface_build/build

# Build the project
RUN cmake .. && make

# Set the default command to run UVface_build++
CMD ["./UVface++"]
