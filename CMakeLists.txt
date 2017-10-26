cmake_minimum_required(VERSION 2.8)

include(PICMake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -mssse3 -msse2")

pi_add_target(GBoW BIN src REQUIRED GSLAM OpenCV)
pi_report_targets()
