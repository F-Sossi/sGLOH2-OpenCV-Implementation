# sGLOH_OpenCV_Implementaion

For this project the objective is to incorporate the sGLOH (scale-invariant Gradient Location and Orientation Histogram) descriptor into the OpenCV environment, employing C++ as our coding language. This project will implement and evaluate the sGLOH descriptor and its performance compared to SIFT (Scale-Invariant Feature Transform) approach, which constitutes our benchmark for assessment.

The Report covering the details of this project is in the main repository titled [CSS587_Computer_Vision_Project_sGLOH2](https://github.com/F-Sossi/sGLOH2-OpenCV-Implementation/blob/master/CSS587_Computer_Vision_Project_sGLOH2.pdf)

Demo Video:  https://www.youtube.com/watch?v=DWoCIwm22bk

# sGLOH_opencv

## Installation

This project requires OpenCV 4.0 with the contributors package installed to be installed on your system. There are two possible scenarios:

1. OpenCV is installed on your main system.
2. OpenCV is installed using vcpkg.

[Instructions to install OpenCV from source](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)

[Instructions to install OpenCV  with vcpkg](https://vcpkg.io/en/)

### If OpenCV is installed on your main system:

You can proceed with the project as is. The CMakeLists.txt file is already set up to find your OpenCV installation.

### If OpenCV is installed using vcpkg:

You need to adjust the toolchain file path in the CMakeLists.txt file to match your vcpkg installation path.

Here's how you can do it:

1. Open the CMakeLists.txt file.
2. Locate the line that sets the CMAKE_TOOLCHAIN_FILE variable.
3. Replace "/home/user/tools/vcpkg/scripts/buildsystems/vcpkg.cmake" with the path to the vcpkg.cmake file in your vcpkg installation.

The updated line should look something like this:

```cmake
set(CMAKE_TOOLCHAIN_FILE "/path/to/your/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
```

Replace "/path/to/your/vcpkg" with the actual path to your vcpkg installation.

After updating the CMakeLists.txt file, you can proceed with the project as usual.

## Building the Project

1. Open a terminal in the project directory.
2. Create a new directory for the build files:

```bash
mkdir build
cd build
```

3. Run CMake to generate the build files:

```bash
cmake ..
```

4. Build the project:

```bash
make
```

This will create an executable named `sGLOH_opencv` in the build directory. You can run it with:

```bash
./sGLOH_opencv
```

Please note that these instructions assume you're using a Unix-like operating system. If you're using Windows, the commands might be slightly different.


 This work is based on the following paper:
 
 Bellavia, Fabio, and Carlo Colombo. "Rethinking the sGLOH descriptor." IEEE Transactions on Pattern Analysis and Machine Intelligence 40.4 (2017): 931-944. 
 [Rethinking the sGLOH descriptor](https://ieeexplore.ieee.org/abstract/document/7911291?casa_token=cPjrrbMCyMEAAAAA:y7ygHc5iHk5N01iA9LYfI_DGw7Y2smUGPRA4fLjWmHsUotf5mr3KL-OS1igD03gTqZlvTYpS)

 
 Bellavia, Fabio, Domenico Tegolo, and Emanuele Trucco. "Improving SIFT-based descriptors stability to rotations." 2010 20th International Conference on Pattern Recognition. IEEE, 2010.
[Improving SIFT-based descriptors stability to rotations](https://ieeexplore.ieee.org/abstract/document/5597547?casa_token=v7UJgCRktggAAAAA:FtfCL8Fr0UTIOseEbSiePHaIYakTIHQgPGZygFYYPXw5Xo-2KjrvKnwiMkAGR_2BTsmrQSY5)

Links: 
View Paper: https://www.overleaf.com/read/hmkghydpbrnm

Test Plan: https://lucid.app/lucidspark/d923e81a-c75c-4dd8-b7c3-e4b6a1192972/edit?invitationId=inv_fbac7c33-fa32-4430-809d-e57e9bfe237c
