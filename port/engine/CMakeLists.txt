# The Strata-compatible engine (`strata-vk`) on ggml's Vulkan backend.
#
# Added to a llama.cpp checkout at Strata's ggml pin (third_party/strata-ggml's UPSTREAM.md: 3cf03257) by
# `-DSTRATA_VK_DIR=<this folder>`; lc/examples/CMakeLists.txt adds this directory.  Building it inside that tree is
# what gives it the llama/ggml targets and the Vulkan shader build without a second, divergent copy of ggml.

set(STRATA_VK_SOURCES
    vk-common.cpp
    vk-options.cpp
    vk-devices.cpp
    vk-engine.cpp
    vk-engine-load.cpp
    vk-mtp.cpp
    vk-rdc.cpp
    vk-serve.cpp
    vk-main.cpp)

set(TARGET strata-vk)
add_executable(${TARGET} ${STRATA_VK_SOURCES})
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
# llama-ext.h (the staging API: llama_set_embeddings_nextn and friends) lives next to the sources, not in include/.
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
if (MSVC)
  target_compile_options(${TARGET} PRIVATE /utf-8)
endif()

# RenderDoc's in-application capture API (STRATA_VK_RDC=<step>).  The header is vendored next to the sources so an
# ordinary build does not need the RenderDoc SDK; the DLL is loaded at run time and a build without it simply reports
# that no capture can be taken.
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(${TARGET} PRIVATE STRATA_HAVE_RDC=1)
if (WIN32)
  target_link_libraries(${TARGET} PRIVATE)
endif()

# The NPU half is compiled into the engine, not built here: ggml-hexagon comes from the llama.cpp tree
# (`-DGGML_HEXAGON=ON` plus HEXAGON_SDK_ROOT).  The engine finds the device at run time and moves the dense tensors
# onto it (vk-devices.cpp), so a tree without the backend is a complete engine that simply reports `npu=none`.

# strata-vision: the picture reader the front end spawns (serve/server.py, class Vision).  It is llama.cpp's mtmd on
# the mmproj file, speaking "ENC <image> <output>" -> "OK <n_tokens>" and writing the .sve record the engine reads on
# its GENI/BGENI line (port/docs/features/VISION.md).  Built only when the tree carries mtmd (LLAMA_BUILD_TOOLS), and
# opt-in at run time: an engine started without --mmproj behaves exactly as before.
# Not "if (TARGET mtmd)": examples/ is configured before tools/, so the target does not exist yet at this point in
# the configure; the source file does, and the link line is resolved after every subdirectory has been added.
if (LLAMA_BUILD_TOOLS AND EXISTS ${CMAKE_SOURCE_DIR}/tools/mtmd/mtmd.cpp)
  add_executable(strata-vision strata-vision.cpp)
  target_link_libraries(strata-vision PRIVATE mtmd llama ${CMAKE_THREAD_LIBS_INIT})
  target_compile_features(strata-vision PRIVATE cxx_std_17)
  if (MSVC)
    target_compile_options(strata-vision PRIVATE /utf-8)
  endif()
  install(TARGETS strata-vision RUNTIME)
else()
  message(STATUS "strata-vision: not built (this llama.cpp tree has no mtmd target; configure with LLAMA_BUILD_TOOLS=ON)")
endif()

# mm-probe: one mul_mat shape, one backend, timed in isolation.  Every kernel decision in this port is measured with it
# (port/docs/ADRENO.md), because the backend's own per-op table brackets more than the kernel.
# requant-tensor: write a companion GGUF with a few tensors requantised, since llama-quantize cannot filter what it
# writes (PERFORMANCE.md section 5.0.1).
add_executable(requant-tensor ${CMAKE_CURRENT_SOURCE_DIR}/../probe/requant-tensor.cpp)
target_link_libraries(requant-tensor PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(requant-tensor PRIVATE cxx_std_17)
if (MSVC)
  target_compile_options(requant-tensor PRIVATE /utf-8)
endif()
add_executable(mm-probe ${CMAKE_CURRENT_SOURCE_DIR}/../probe/mm-probe.cpp)
target_link_libraries(mm-probe PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
# renderdoc_app.h for the one-dispatch capture (STRATA_VK_RDC_PROBE), vendored next to the engine sources
target_include_directories(mm-probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(mm-probe PRIVATE cxx_std_17)
if (MSVC)
  target_compile_options(mm-probe PRIVATE /utf-8)
endif()


# codec-probe: quantise one model-like row to a type and dequantise it again, to measure the codec against the f32 truth.
# It links llama rather than ggml because ggml_quantize_chunk never returns for the K-quants and the trellis types in a
# process that has not run llama_backend_init (port/docs/PORT-STATE.md records the measurement).
add_executable(codec-probe ${CMAKE_CURRENT_SOURCE_DIR}/../probe/codec-probe.cpp)
target_link_libraries(codec-probe PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(codec-probe PRIVATE cxx_std_17)
if (MSVC)
  target_compile_options(codec-probe PRIVATE /utf-8)
endif()
