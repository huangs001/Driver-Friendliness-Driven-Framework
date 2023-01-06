cmake_minimum_required (VERSION 3.1)
project (FRIENDLY_ROUTE)

SET(CMAKE_BUILD_TYPE "Release")

find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

file(GLOB src ${PROJECT_SOURCE_DIR}/src/*.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable (HT_process ${src})
target_link_libraries(HT_process PRIVATE Threads::Threads)
target_compile_features(HT_process  PRIVATE cxx_std_17)
