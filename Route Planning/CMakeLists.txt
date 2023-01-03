cmake_minimum_required (VERSION 3.1)
project (FRIENDLY_ROUTE)

find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

set(THIRDPARTY_DIR ${PROJECT_SOURCE_DIR}/thirdparty)
file(GLOB routingkit_src ${THIRDPARTY_DIR}/RoutingKit/src/*.cpp)
add_library (RoutingKit ${routingkit_src})
target_include_directories (RoutingKit PUBLIC ${THIRDPARTY_DIR}/RoutingKit/include)
target_link_libraries(RoutingKit PRIVATE ZLIB::ZLIB)

include_directories(${THIRDPARTY_DIR})

file(GLOB src_list ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_executable (DFNav_demo ${src_list})
target_link_libraries(DFNav_demo PRIVATE Threads::Threads)
target_link_libraries(DFNav_demo LINK_PUBLIC RoutingKit)
