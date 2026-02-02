# cmake_minimum_required(VERSION 3.25.0)
# project(train_gpt2cu LANGUAGES C CXX CUDA)

# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CUDA_STANDARD 20)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
# set(CMAKE_CUDA_ARCHITECTURES "native")

# include(cmake/CPM.cmake)
# CPMAddPackage("gh:NVIDIA/cccl#main")
# CPMAddPackage("gh:NVIDIA/nvbench#main")

# find_package(OpenMP REQUIRED)

# add_executable(train_gpt2 train_gpt2.cpp)
# target_link_libraries(train_gpt2 PRIVATE OpenMP::OpenMP_CXX)

# find_package(CUDAToolkit)
# add_executable(train_gpt2cu train_gpt2.cu)
# target_link_libraries(train_gpt2cu CCCL::CCCL CUDA::cublas CUDA::cublasLt)
# target_compile_options(train_gpt2cu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --extended-lambda>)

# add_subdirectory(benches)

cmake_minimum_required(VERSION 3.25.0)
project(train_gpt2cu LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
set(CMAKE_CUDA_ARCHITECTURES "native")

# include(cmake/CPM.cmake)
# CPMAddPackage("gh:NVIDIA/cccl#main")
# CPMAddPackage("gh:NVIDIA/nvbench#main")

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit)

# =========================================================
# 1. 批量处理 CPU 文件 (.cpp)
# =========================================================
set(CPU_SOURCES
    train_gpt2.cpp
    train_gpt2_by_myself.cpp
)

foreach(src_file ${CPU_SOURCES})
    get_filename_component(target_name ${src_file} NAME_WE)
    add_executable(${target_name} ${src_file})
    # CPU 专用的库
    target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
    # 【新增】CPU 性能优化选项 (非常重要！！！)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target_name} PRIVATE 
            -O3              # 开启最高级优化
            -march=native    # 针对你的 i7-13700F 开启 AVX2/AVX512 指令集
            -ffast-math      # 允许浮点运算牺牲微小精度换取速度
            -fopenmp         # 显式加上 OpenMP 编译标志 (防止链接库时漏掉)
        )
    endif()
endforeach()

# # =========================================================
# # 2. 批量处理 CUDA 文件 (.cu)
# # =========================================================
# set(CUDA_SOURCES
#     train_gpt2.cu
#     train_gpt2_by_myself.cu 
# )

# foreach(src_file ${CUDA_SOURCES})
#     get_filename_component(target_name ${src_file} NAME_WE)
    
#     # 【修改点 1】给可执行文件改名！
#     # 比如原来叫 train_gpt2，现在叫 train_gpt2_cu_exe 或者 train_gpt2cu
#     # 这样就不会覆盖 train_gpt2.cu 源代码了
#     set(exe_name "${target_name}_cu_exe") 

#     add_executable(${exe_name} ${src_file})
    
#     # 【修改点 2】后面引用也要改用新的名字
#     target_link_libraries(${exe_name} CCCL::CCCL CUDA::cublas CUDA::cublasLt)
    
#     target_compile_options(${exe_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --extended-lambda>)
# endforeach()

# add_subdirectory(benches)