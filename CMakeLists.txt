cmake_minimum_required (VERSION 2.8)
if(APPLE)
    SET(CMAKE_CXX_COMPILER "g++-5")
    SET(CMAKE_C_COMPILER "gcc-5")
endif()
project (ftrl)

OPTION(USE_MPI "MPI based parallel learning" OFF)
OPTION(USE_OPENMP "Enable OpenMP" ON)
OPTION(USE_GPU "Enable GPU-acclerated training (EXPERIMENTAL)" OFF)

if(APPLE)
    OPTION(APPLE_OUTPUT_DYLIB "Output dylib shared library" OFF)
endif()

if(USE_MPI)
    find_package(MPI REQUIRED)
    ADD_DEFINITIONS(-DUSE_MPI)
    MESSAGE(${MPI_LIBRARIES})
    MESSAGE(${MPI_CXX_LIBRARIES})
else()
    ADD_DEFINITIONS(-DUSE_SOCKET)
endif(USE_MPI)

if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    # SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    # Ignore unknown #pragma warning
    if( (CMAKE_CXX_COMPILER_ID MATCHES "[cC][lL][aA][nN][gG]")
      OR (CMAKE_CXX_COMPILER_ID MATCHES "[gG][nN][uU]"))
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    endif()
endif(USE_OPENMP)

if(UNIX OR MINGW OR CYGWIN)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -O3 -Wextra -Wall -Wno-ignored-attributes -Wno-unknown-pragmas")
endif()

add_executable(ftrl ftrl.cpp)

target_link_libraries(ftrl pthread rt)
