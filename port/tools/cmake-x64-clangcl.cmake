# Cross-build for x86_64 Windows on this ARM64 host, so the engine can run under Prism and be profiled by the x64
# RenderDoc.  CMAKE_SYSTEM_PROCESSOR has to be spelled out: without it CMake reports the host as ARM64 and ggml picks
# the NEON sources (duplicate symbols at link time), and the Vulkan SDK resolves to its ARM64 import library.
set(CMAKE_SYSTEM_NAME      Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(_vs "C:/Program Files/Microsoft Visual Studio/18/Community")
set(CMAKE_C_COMPILER   "${_vs}/VC/Tools/Llvm/ARM64/bin/clang-cl.exe")
set(CMAKE_CXX_COMPILER "${_vs}/VC/Tools/Llvm/ARM64/bin/clang-cl.exe")

set(CMAKE_C_FLAGS_INIT   "--target=x86_64-pc-windows-msvc")
set(CMAKE_CXX_FLAGS_INIT "--target=x86_64-pc-windows-msvc")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
