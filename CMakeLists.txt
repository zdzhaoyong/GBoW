cmake_minimum_required(VERSION 2.8)

include(PICMake)
pi_add_target(GBoW BIN src REQUIRED GSLAM)
pi_report_targets()
