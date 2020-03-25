cmake_minimum_required(VERSION 3.12)
project(Particles)

set(CMAKE_CXX_STANDARD 17)

set(INTEL_CC ON)

if (INTEL_CC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp -Wl,-rpath=/opt/intel/compilers_and_libraries/linux/lib/intel64 -O3 -xHost -ipo -no-prec-div")
    set(CMAKE_CXX_COMPILER "/opt/intel/compilers_and_libraries/linux/bin/intel64/icc")
else (INTEL_CC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fopenmp")
endif (INTEL_CC)

add_executable(Particles src/main.cpp src/SystemParticles.cpp)