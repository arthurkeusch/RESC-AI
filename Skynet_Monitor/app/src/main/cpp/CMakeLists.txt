cmake_minimum_required(VERSION 3.22.1)

project("llama_android")

# Utilisation du C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Dossier llama.cpp
set(LLAMA_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)

# Désactiver tous les backends non ARM pour Android CPU
add_definitions(
        -DGGML_USE_CPU
        -DGGML_NO_ACCELERATE
        -DGGML_NO_AVX
        -DGGML_NO_AVX2
        -DGGML_NO_AVX512
        -DGGML_NO_FMA
        -DGGML_NO_SSE
        -DGGML_NO_SSE2
        -DGGML_NO_SSE3
        -DGGML_NO_SSSE3
        -DGGML_NO_SSE4_1
        -DGGML_NO_SSE4_2
        -DGGML_NO_F16C
        -DGGML_NO_RISCV
        -DGGML_NO_WASM
        -DGGML_NO_LOONGARCH
)

# Sélection des fichiers nécessaires
file(GLOB_RECURSE LLAMA_SOURCES
        ${LLAMA_CPP_DIR}/ggml.c
        ${LLAMA_CPP_DIR}/ggml-alloc.c
        ${LLAMA_CPP_DIR}/ggml-backend.c
        ${LLAMA_CPP_DIR}/ggml-quants.c
        ${LLAMA_CPP_DIR}/llama.cpp
        ${LLAMA_CPP_DIR}/unicode.cpp
)

# Fichier JNI de ton projet
set(PROJECT_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/native-lib.cpp
)

# Créer la bibliothèque native
add_library(
        llama
        SHARED
        ${LLAMA_SOURCES}
        ${PROJECT_SOURCES}
)

# Inclure les fichiers headers
target_include_directories(llama PUBLIC
        ${LLAMA_CPP_DIR}
)

# Lier bibliothèque Android log
find_library(log-lib log)
target_link_libraries(llama ${log-lib})
