cmake_minimum_required(VERSION 2.8)  

project(TestProject)    
FIND_PACKAGE(Qt4 REQUIRED)
INCLUDE(${QT_USE_FILE})




find_package( OpenCV REQUIRED )


#This is required in order to use openMP
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()



include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
add_definitions(-DFOO)
# create an executable and a library target, both requiring automoc:


#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 
add_library(mylib STATIC interfazPrincipal.cpp plotSparseSolution.cpp qcustomplot.cpp guiOtherConfigurations.cpp guiConfigDetector.cpp detector.cpp trackerWindows_gui.cpp trackerWindows.cpp guiFaceRecognizer.cpp dataBaseImages.cpp dictionary.cpp recognizerFacial.cpp descriptor.cpp gtp2.cpp)

#set(CMAKE_BUILD_TYPE Release -D)
set(CMAKE_BUILD_TYPE Release)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3")
#set(CMAKE_CXX_FLAGS "-O2")        ## Optimize
#set(CMAKE_EXE_LINKER_FLAGS "-s")  ## Strip binary

add_executable(UVface++ main.cpp)
set_target_properties(UVface++ mylib  PROPERTIES AUTOMOC TRUE)
target_link_libraries(UVface++ mylib -fopenmp ${OpenCV_LIBS} ${QT_LIBRARIES})



