cmake_minimum_required( VERSION 2.8 )

set( CMAKE_CXX_FLAGS "-std=c++11 -Wall -O2 -march=native" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package( OpenCV REQUIRED )
find_package(DLib REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS} 
    ${DLib_INCLUDE_DIRS}
    )
add_executable( bow_sift BoFSIFT.cpp )
target_link_libraries( bow_sift 
    ${OpenCV_LIBS}
    )

add_executable( DBoW2_demo DBoW2_demo.cpp )
target_link_libraries( DBoW2_demo 
    DBoW2
    ${OpenCV_LIBS}
    ${DLib_LIBS}
    )

