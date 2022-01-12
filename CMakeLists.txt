cmake_minimum_required(VERSION 2.8)
project(RCLC)
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

add_executable(CompareTest apps/CompareTest.cpp src/global.cpp src/GMMFit.cpp src/GaussianFunction.cpp include/opt/board_fitting_cost.h)
target_link_libraries(CompareTest  ${PCL_LIBRARIES}
        /usr/lib/x86_64-linux-gnu/liblz4.a /usr/lib/x86_64-linux-gnu/liblz4.so
        ${CERES_LIBRARIES} ${OpenCV_LIBS})

add_executable(CodeTest apps/CodeTest.cpp src/global.cpp src/GMMFit.cpp src/GaussianFunction.cpp include/opt/board_fitting_cost.h)
target_link_libraries(CodeTest  ${PCL_LIBRARIES}
        /usr/lib/x86_64-linux-gnu/liblz4.a /usr/lib/x86_64-linux-gnu/liblz4.so
        ${CERES_LIBRARIES} ${OpenCV_LIBS})

add_executable(BoardSegmentation apps/BoardSegmentation.cpp src/global.cpp src/GMMFit.cpp src/GaussianFunction.cpp include/opt/board_fitting_cost.h)
target_link_libraries(BoardSegmentation  ${PCL_LIBRARIES}
        /usr/lib/x86_64-linux-gnu/liblz4.a /usr/lib/x86_64-linux-gnu/liblz4.so
        ${CERES_LIBRARIES} ${OpenCV_LIBS})


#单纯使用分割出来的标定板进行角点估计
add_executable(BoardPointExt apps/BoardPointExt.cpp src/global.cpp src/GMMFit.cpp src/GaussianFunction.cpp include/opt/board_fitting_cost.h)
target_link_libraries(BoardPointExt  ${PCL_LIBRARIES}
        /usr/lib/x86_64-linux-gnu/liblz4.a /usr/lib/x86_64-linux-gnu/liblz4.so
        ${CERES_LIBRARIES} ${OpenCV_LIBS})

#使用分割出来的标定板进行角点估计之后，把结果写入到文件并且使用多帧进行优化，每次测试20种不同的组合
add_executable(EvaluateSim apps/EvaluateSim.cpp src/global.cpp src/GMMFit.cpp src/GaussianFunction.cpp include/opt/board_fitting_cost.h)
target_link_libraries(EvaluateSim  ${PCL_LIBRARIES}
        /usr/lib/x86_64-linux-gnu/liblz4.a /usr/lib/x86_64-linux-gnu/liblz4.so
        ${CERES_LIBRARIES} ${OpenCV_LIBS})

#在真实数据上进行测试，使用分割出来的标定板进行角点估计之后，求解重投影误差
add_executable(EvaluateReal apps/EvaluateReal.cpp src/global.cpp src/GMMFit.cpp src/GaussianFunction.cpp include/opt/board_fitting_cost.h)
target_link_libraries(EvaluateReal  ${PCL_LIBRARIES}
        /usr/lib/x86_64-linux-gnu/liblz4.a /usr/lib/x86_64-linux-gnu/liblz4.so
        ${CERES_LIBRARIES} ${OpenCV_LIBS})

#在真实数据上进行测试，使用分割出来的标定板进行角点估计之后，求解重投影误差
add_executable(CrossEvaluate apps/CrossEvaluate.cpp src/global.cpp src/GMMFit.cpp src/GaussianFunction.cpp include/opt/board_fitting_cost.h)
target_link_libraries(CrossEvaluate  ${PCL_LIBRARIES}
        /usr/lib/x86_64-linux-gnu/liblz4.a /usr/lib/x86_64-linux-gnu/liblz4.so
        ${CERES_LIBRARIES} ${OpenCV_LIBS})