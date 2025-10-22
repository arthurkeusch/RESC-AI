# app/src/main/cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.22)
project(skynet_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Alléger la build llama.cpp pour Android
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)

# Chemin vers le dossier source de llama.cpp (surchargeable depuis Gradle via -DLLAMA_DIR=...)
if(NOT DEFINED LLAMA_DIR)
    set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp CACHE PATH "Path to llama.cpp")
endif()

# Importe le projet CMake de llama.cpp
add_subdirectory(${LLAMA_DIR})

# Votre lib JNI
add_library(native-lib SHARED
        native-lib.cpp
)

# Headers publics de llama.cpp -> permet #include "llama.h"
target_include_directories(native-lib PRIVATE
        ${LLAMA_DIR}/include
)

# Lien contre la cible 'llama' et la lib 'log' Android
target_link_libraries(native-lib PRIVATE
        llama
        log
)

# ---------------------------------------------------------
# app/src/main/cpp/native-lib.cpp (seulement la ligne d'include à corriger)
# Remplacez:
#   #include "llama.cpp/llama.h"
# Par:
#   #include "llama.h"
