cmake_minimum_required(VERSION 4.4)
project(AIMLDL)

set(CMAKE_CXX_STANDARD 14)

add_executable(AIMLDL
#        chineseSpy/intro.cpp
        chineseSpy/Matrix.cpp
        chineseSpy/main.h
        chineseSpy/Matrix.hpp
        chineseSpy/main.cpp)

add_executable(INTRO
        chineseSpy/intro.cpp
)