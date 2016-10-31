cmake_minimum_required( VERSION 2.8 )

set( CMAKE_CXX_FLAGS "-std=c++11 -Wall -O2 -march=native" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package( OpenCV REQUIRED )

add_executable( bow_sift BoFSIFT.cpp )
target_link_libraries( bow_sift 
    ${OpenCV_LIBS}
    )


