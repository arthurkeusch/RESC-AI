cmake_minimum_required(VERSION 3.22.1)
project("llama-android")

include(FetchContent)

# --- Configuration de llama.cpp ---
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(GGML_NATIVE_DEFAULT OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
        llama
        GIT_REPOSITORY https://github.com/ggml-org/llama.cpp
        GIT_TAG master
)

FetchContent_MakeAvailable(llama)

# --- Librairie Android ---
add_library(${CMAKE_PROJECT_NAME} SHARED
        llama-android.cpp
)

target_link_libraries(${CMAKE_PROJECT_NAME}
        llama
        common
        android
        log
)
