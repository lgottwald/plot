cmake_minimum_required(VERSION 2.8)

PROJECT(plot)

include(ExternalProject)

set(LOCAL_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install )
set(LOCAL_PREFIX_PATH ${CMAKE_INSTALL_PREFIX} ${LOCAL_INSTALL_PREFIX} ${CMAKE_PREFIX_PATH})
string(REPLACE ";" "^^" LOCAL_PREFIX_PATH "${LOCAL_PREFIX_PATH}")

ExternalProject_Add (
  plot
  LIST_SEPARATOR ^^
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src
  UPDATE_COMMAND ""
  INSTALL_COMMAND make install > ${LOCAL_INSTALL_PREFIX}/install_output.log
  INSTALL_DIR ${LOCAL_INSTALL_PREFIX}
  CMAKE_ARGS -DCMAKE_PREFIX_PATH=${LOCAL_PREFIX_PATH} -DCMAKE_INSTALL_PREFIX=${LOCAL_INSTALL_PREFIX} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  )

ExternalProject_Add_Step(plot forceconfigure
   COMMAND ${CMAKE_COMMAND} -E echo "Force configure of plot"
   DEPENDEES update
   DEPENDERS configure
   ALWAYS 1)

find_package( libspline QUIET )
if( NOT libspline_FOUND )
  ExternalProject_Add(
    libspline
    LIST_SEPARATOR ^^
    GIT_REPOSITORY https://github.com/rgottwald/spline.git
    GIT_TAG origin/master
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
    CMAKE_ARGS -DCMAKE_PREFIX_PATH=${LOCAL_PREFIX_PATH} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  )
  add_dependencies( plot libspline )
endif()

find_package( cpplsq QUIET )
if( NOT cpplsq_FOUND )
 ExternalProject_Add(
  libcpplsq
  LIST_SEPARATOR ^^
  GIT_REPOSITORY https://github.com/rgottwald/cpplsq.git
  GIT_TAG origin/master
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
  CMAKE_ARGS -DCMAKE_PREFIX_PATH=${LOCAL_PREFIX_PATH} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
 )
 add_dependencies( plot libcpplsq )
endif()

find_package( libsdo QUIET )
if( NOT libsdo_FOUND )
ExternalProject_Add(
  libsdo
  LIST_SEPARATOR ^^
  GIT_REPOSITORY https://github.com/rgottwald/libsdo.git
  GIT_TAG origin/master
  INSTALL_COMMAND ""
  CMAKE_ARGS -DCMAKE_PREFIX_PATH=${LOCAL_PREFIX_PATH} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
)
add_dependencies(plot libsdo)
endif()

INSTALL( PROGRAMS ${LOCAL_INSTALL_PREFIX}/bin/plot DESTINATION bin )