cmake_minimum_required(VERSION 4.3)
project(mako_image LANGUAGES CXX)

if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    message(FATAL_ERROR "mako-image targets Apple Silicon only")
endif()

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OSX_DEPLOYMENT_TARGET 26.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
    message(STATUS "CMAKE_BUILD_TYPE not set: using RelWithDebInfo")
endif()

add_compile_options(-Wall -Wextra -Wpedantic)
# Warnings fail the build (C++ and Metal); cmake --compile-no-warning-error turns that off for one configure.
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/mako-common/CMakeLists.txt)
    message(FATAL_ERROR "external/mako-common is missing: git submodule update --init")
endif()
add_subdirectory(external/mako-common)

set(S ${CMAKE_CURRENT_SOURCE_DIR}/src)
mako_add_kernels(image_kernels SOURCES ${S}/kernels/image.metal)

add_library(mako_imagegen STATIC ${S}/image/engine.cpp ${S}/image/qwen_image.cpp ${S}/image/krea2.cpp
            ${S}/image/moss_image.cpp ${S}/image/anima.cpp ${S}/image/sdxl.cpp ${S}/image/clip_tokenizer.cpp
            ${S}/image/png.cpp)
target_include_directories(mako_imagegen PUBLIC ${S})
target_link_libraries(mako_imagegen PUBLIC mako_gpu mako_safetensors mako_text mako_log
                      PRIVATE glaze_headers "-framework ImageIO" "-framework CoreGraphics" "-framework CoreFoundation")

add_library(mako_image_server STATIC ${S}/server/image_models.cpp ${S}/server/image_server.cpp)
target_include_directories(mako_image_server PUBLIC ${S})
target_link_libraries(mako_image_server PUBLIC mako_imagegen mako_memory mako_http PRIVATE glaze_headers httplib_headers)

add_executable(mako_image tools/mako_image.cpp)
target_link_libraries(mako_image PRIVATE mako_imagegen mako_memory mako_app)
add_executable(mako_image_serve tools/mako_image_serve.cpp)
target_link_libraries(mako_image_serve PRIVATE mako_image_server mako_app)

# Model-backed tests need the models' directories and the references' tensors; without them they report as skipped.
set(MAKO_QWEN_IMAGE "$ENV{MAKO_QWEN_IMAGE}" CACHE PATH "Qwen-Image 2.1 directory (its safetensors) for its tests")
set(MAKO_QWEN_IMAGE_REF "$ENV{MAKO_QWEN_IMAGE_REF}" CACHE PATH "Qwen-Image 2.1 reference tensors for its tests")
set(MAKO_KREA2 "$ENV{MAKO_KREA2}" CACHE PATH "Krea 2 directory (its safetensors) for its tests")
set(MAKO_KREA2_TRANSFORMER "$ENV{MAKO_KREA2_TRANSFORMER}" CACHE FILEPATH "Krea 2 transformer checkpoint for its tests")
set(MAKO_KREA2_REF "$ENV{MAKO_KREA2_REF}" CACHE PATH "Krea 2 reference tensors for its tests")
set(MAKO_MOSS "$ENV{MAKO_MOSS}" CACHE PATH "MOSS-Image directory (its safetensors) for its tests")
set(MAKO_MOSS_REF "$ENV{MAKO_MOSS_REF}" CACHE PATH "MOSS-Image reference tensors for its tests")
set(MAKO_ANIMA "$ENV{MAKO_ANIMA}" CACHE PATH "Anima directory (its split files) for its tests")
set(MAKO_ANIMA_TRANSFORMER "$ENV{MAKO_ANIMA_TRANSFORMER}" CACHE FILEPATH "Anima transformer file for its tests")
set(MAKO_ANIMA_REF "$ENV{MAKO_ANIMA_REF}" CACHE PATH "Anima reference tensors for its tests")
set(MAKO_SDXL "$ENV{MAKO_SDXL}" CACHE PATH "SDXL directory (its checkpoint and tokenizer/) for its tests")
set(MAKO_SDXL_REF "$ENV{MAKO_SDXL_REF}" CACHE PATH "SDXL reference tensors for its tests")
set(MAKO_SDXL_LORA "$ENV{MAKO_SDXL_LORA}" CACHE FILEPATH "SDXL LoRA (Kohya's format) its reference's lora_manifest.json names")
set(MAKO_T5_TOKENIZER "$ENV{MAKO_T5_TOKENIZER}" CACHE FILEPATH "T5 tokenizer.json for the unigram tokenizer test")
enable_testing()
add_subdirectory(tests)
