cmake_minimum_required(VERSION 2.8)

find_package(cpplsq REQUIRED)
find_package(libspline REQUIRED)
find_package(libsdo REQUIRED)
find_package(OpenMP)
if(OPENMP_FOUND)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

INCLUDE_DIRECTORIES(
  ${cpplsq_INCLUDE_DIRS}
  ${libspline_INCLUDE_DIRS}
  ${libsdo_INCLUDE_DIRS}
)

add_definitions(
  ${cpplsq_DEFINITIONS}
)

ADD_EXECUTABLE(plot
        plot.cpp
        )

TARGET_LINK_LIBRARIES( plot ${cpplsq_LIBRARIES} ${libsdo_LIBRARIES} )        
set_target_properties(plot PROPERTIES COMPILE_FLAGS "-std=c++11 -pedantic-errors -Wall -Wextra -Wno-unused-parameter -Wno-unused-local-typedefs")

INSTALL(TARGETS plot RUNTIME DESTINATION bin)