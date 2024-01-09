
cmake_minimum_required(VERSION 3.0)
project(stats CXX)

# 设置编译为Release模式
set(CMAKE_BUILD_TYPE "Release")

# 设置C/C++标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # 设置编译得到的文件的保存目录为 mbapy/storage
# set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/../../storage)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")

# 定义动态链接库目标
add_library(stats SHARED main.cpp)

# # 安装动态链接库到指定目录 mbapy/storage
# install(TARGETS stats LIBRARY DESTINATION ../../storage)


