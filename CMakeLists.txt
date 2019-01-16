cmake_minimum_required(VERSION 3.3)
project(openmp_samples)

set(CMAKE_CXX_STANDARD 11)

# try also to compile and execute without: -fopenmp
# in Linux just use -fopenmp, without -lomp and -Xpreprocessor
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")

include_directories( BEFORE ${MY_SOURCE_DIR} )
set(SOURCE_FILES main.cpp)
add_executable(mean_shift main.cpp csvio.cpp csvio.h)
# set(SOURCE_FILES main.cpp generator.cpp PPM.h stb_image_write.h Image.h stb_image.h)
# add_executable(mean_shift main.cpp PPM.h PPM.cpp stb_image_write.h stb_image.cpp stb_image_write.cpp Image.h Image.cpp csvio.cpp csvio.h)
# add_executable(generator generator.cpp csvio.cpp csvio.h)