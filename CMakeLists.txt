cmake_minimum_required(VERSION 2.8)
project(RCLC_v2)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

#LIBS
set(OpenCV_DIR /home/lab/cpplib/opencv430/opencv/build)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(PCL REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

include_directories(
        ${Boost_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        include
        include/opt
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_directories(${Boost_LIBRARY_DIRS})


FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

add_executable(BoardSegmentation apps/BoardSegmentation.cpp src/global.cpp src/GMMFit.cpp src/GaussianFunction.cpp include/opt/board_fitting_cost.h)
target_link_libraries(BoardSegmentation  ${PCL_LIBRARIES}
        /usr/lib/x86_64-linux-gnu/liblz4.a /usr/lib/x86_64-linux-gnu/liblz4.so
        ${CERES_LIBRARIES} ${OpenCV_LIBS})

add_executable(BoardPointExt apps/BoardPointExt.cpp src/global.cpp src/GMMFit.cpp src/GaussianFunction.cpp include/opt/board_fitting_cost.h)
target_link_libraries(BoardPointExt  ${PCL_LIBRARIES}
        /usr/lib/x86_64-linux-gnu/liblz4.a /usr/lib/x86_64-linux-gnu/liblz4.so
        ${CERES_LIBRARIES} ${OpenCV_LIBS})

add_executable(Calibrate apps/Calibrate.cpp src/global.cpp src/GMMFit.cpp src/GaussianFunction.cpp include/opt/board_fitting_cost.h)
target_link_libraries(Calibrate  ${PCL_LIBRARIES}
        /usr/lib/x86_64-linux-gnu/liblz4.a /usr/lib/x86_64-linux-gnu/liblz4.so
        ${CERES_LIBRARIES} ${OpenCV_LIBS})
