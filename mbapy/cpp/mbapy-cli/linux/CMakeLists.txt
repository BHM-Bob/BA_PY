
cmake_minimum_required(VERSION 3.0)
project(mbapy_cli CXX)

# 设置编译为Release模式
set(CMAKE_BUILD_TYPE "Release")

# 设置编译得到的可执行文件的保存目录为 mbapy/build
set(CMAKE_BINARY_DIR ${PROJECT_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")

# 添加源代码文件
add_executable(mbapy-cli main.cpp)

# 针对不同平台进行条件编译
if(WIN32)
    # 针对 Windows 平台进行配置
    # ...
else()
    # 针对 Linux 平台进行配置
    # ...
endif()
