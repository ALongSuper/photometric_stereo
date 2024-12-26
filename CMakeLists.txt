cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)
PROJECT(photometric_stereo LANGUAGES CXX CUDA)
 
find_package(CUDA)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# FIND_PACKAGE(PCL REQUIRED)
# include_directories(${PCL_INCLUDE_DIRS})
# link_directories(${PCL_LIBRARY_DIRS})
# add_definitions(${PCL_DEFINITIONS})
# link_libraries(${PCL_LIBRARIES})

ADD_DEFINITIONS(-std=c++17)

FILE(GLOB SOURCES src/*.cpp)
FILE(GLOB HEADERS src/*.h)

ADD_EXECUTABLE(main ${SOURCES} ${HEADERS})


TARGET_LINK_LIBRARIES(main ${OpenCV_LIBS} ${TORCH_LIBRARIES} )
# TARGET_LINK_LIBRARIES(main ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${PCL_LIBRARIES})



