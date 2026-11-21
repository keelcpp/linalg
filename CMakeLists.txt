cmake_minimum_required(VERSION 3.20)
project(keel_linalg VERSION 0.1.0 LANGUAGES CXX)

option(KEEL_LINALG_WITH_HIGHWAY "Enable Google Highway SIMD acceleration in keel_linalg" ON)
option(KEEL_LINALG_WITH_THREADS "Enable multi-threaded parallel execution via keel::ThreadPool" ON)
option(KEEL_LINALG_NO_EXCEPTIONS_RTTI "Compile keel_linalg without C++ exceptions and RTTI" OFF)
option(KEEL_LINALG_BUILD_TESTS "Build keel_linalg unit tests and header purity checks" ${PROJECT_IS_TOP_LEVEL})

if(NOT TARGET keel::keel)
    find_package(keel QUIET CONFIG)
endif()
if(NOT TARGET keel::keel)
    include(FetchContent)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../keel/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL)
        set(FETCHCONTENT_SOURCE_DIR_KEEL "${CMAKE_CURRENT_SOURCE_DIR}/../keel" CACHE PATH "Local keel checkout")
    endif()
    set(KEEL_NO_EXCEPTIONS_RTTI ${KEEL_LINALG_NO_EXCEPTIONS_RTTI} CACHE BOOL "" FORCE)
    FetchContent_Declare(
        keel
        GIT_REPOSITORY https://github.com/keelcpp/keel.git
        GIT_TAG        f9fcebfba18d98d2c14c8e62a1913412f1299220
    )
    FetchContent_MakeAvailable(keel)
endif()

if(KEEL_LINALG_WITH_HIGHWAY)
    if(NOT TARGET hwy)
        find_package(hwy QUIET CONFIG)
        if(TARGET hwy::hwy AND NOT TARGET hwy)
            add_library(hwy ALIAS hwy::hwy)
        endif()
    endif()
    if(NOT TARGET hwy)
        include(FetchContent)
        if(NOT DEFINED FETCHCONTENT_SOURCE_DIR_HIGHWAY)
            foreach(_hwy_cand
                "${CMAKE_CURRENT_SOURCE_DIR}/../../numkit/bundle/build/_deps/highway-src"
                "${CMAKE_CURRENT_SOURCE_DIR}/../../numkit/ops/build/_deps/highway-src"
            )
                if(EXISTS "${_hwy_cand}/CMakeLists.txt")
                    set(FETCHCONTENT_SOURCE_DIR_HIGHWAY "${_hwy_cand}" CACHE PATH "Local Highway checkout")
                    break()
                endif()
            endforeach()
        endif()
        set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
        set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
        set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
        set(_keel_linalg_saved_warn_deprecated "${CMAKE_WARN_DEPRECATED}")
        set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            highway
            GIT_REPOSITORY https://github.com/google/highway.git
            GIT_TAG        1.2.0
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(highway)
        if(DEFINED _keel_linalg_saved_warn_deprecated AND NOT "${_keel_linalg_saved_warn_deprecated}" STREQUAL "")
            set(CMAKE_WARN_DEPRECATED "${_keel_linalg_saved_warn_deprecated}" CACHE BOOL "" FORCE)
        endif()
    endif()
endif()

set(KEEL_LINALG_SOURCES
    src/blas_common.cpp
    src/solve.cpp
    src/decompositions.cpp
    src/factor_updates.cpp
    src/eig_schur.cpp
    src/qz.cpp
    src/matrix_functions.cpp
    src/special_solvers.cpp
)

if(KEEL_LINALG_WITH_HIGHWAY)
    list(APPEND KEEL_LINALG_SOURCES
        src/blas1_highway.cpp
        src/blas2_highway.cpp
        src/blas3_highway.cpp
    )
else()
    list(APPEND KEEL_LINALG_SOURCES
        src/blas1_portable.cpp
        src/blas2_portable.cpp
        src/blas3_portable.cpp
    )
endif()

add_library(keel_linalg STATIC ${KEEL_LINALG_SOURCES})
add_library(keel::linalg ALIAS keel_linalg)

target_link_libraries(keel_linalg PUBLIC keel::keel)
if(KEEL_LINALG_WITH_HIGHWAY)
    target_link_libraries(keel_linalg PRIVATE $<BUILD_INTERFACE:hwy>)
    target_compile_definitions(keel_linalg PRIVATE KEEL_LINALG_WITH_HIGHWAY=1)
    get_target_property(_keel_linalg_hwy_inc hwy INTERFACE_INCLUDE_DIRECTORIES)
    if(_keel_linalg_hwy_inc)
        target_include_directories(keel_linalg SYSTEM PRIVATE ${_keel_linalg_hwy_inc})
    endif()
endif()

if(KEEL_LINALG_WITH_THREADS)
    target_compile_definitions(keel_linalg PRIVATE KEEL_LINALG_WITH_THREADS=1)
endif()

target_compile_features(keel_linalg PUBLIC cxx_std_20)
set_target_properties(keel_linalg PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(keel_linalg
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

if(MSVC)
    target_compile_options(keel_linalg PRIVATE /W4 /utf-8 /bigobj)
    if(KEEL_LINALG_NO_EXCEPTIONS_RTTI)
        string(REGEX REPLACE "/EH[a-z]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REGEX REPLACE "/GR" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        target_compile_options(keel_linalg PUBLIC /EHs-c- /GR-)
        target_compile_definitions(keel_linalg PUBLIC _HAS_EXCEPTIONS=0)
    endif()
else()
    target_compile_options(keel_linalg PRIVATE -Wall -Wextra -Wpedantic)
    if(KEEL_LINALG_NO_EXCEPTIONS_RTTI)
        target_compile_options(keel_linalg PUBLIC -fno-exceptions -fno-rtti)
    endif()
endif()

if(COMMAND keel_enforce_guidelines)
    keel_enforce_guidelines(keel_linalg
        SCOPE ALL
        BAN_RAW_INTS ALL
        HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ALLOWED_ANGLE_PREFIXES "hwy/"
        STANDALONE_HEADERS ${KEEL_LINALG_BUILD_TESTS}
        TEST_NAME test_guidelines_headers
        WARNINGS OFF
    )
endif()

include(CMakePackageConfigHelpers)
install(TARGETS keel_linalg EXPORT keel_linalgTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/keel DESTINATION include)
install(EXPORT keel_linalgTargets
    FILE keel_linalgTargets.cmake
    NAMESPACE keel::
    DESTINATION lib/cmake/keel_linalg
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/keel_linalgConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/keel_linalgConfig.cmake"
    INSTALL_DESTINATION lib/cmake/keel_linalg
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/keel_linalgConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/keel_linalgConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/keel_linalgConfigVersion.cmake"
    DESTINATION lib/cmake/keel_linalg
)

if(KEEL_LINALG_BUILD_TESTS)
    enable_testing()

    add_executable(test_linalg tests/test_linalg.cpp)
    target_link_libraries(test_linalg PRIVATE keel::linalg)
    if(MSVC)
        target_compile_options(test_linalg PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_linalg PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME test_linalg COMMAND test_linalg)

    get_target_property(KEEL_CORE_INC_DIRS keel::keel INTERFACE_INCLUDE_DIRECTORIES)
    set(KEEL_CORE_INC_CLEAN "")
    foreach(IDIR IN LISTS KEEL_CORE_INC_DIRS)
        string(REGEX REPLACE "\\$<BUILD_INTERFACE:([^>]+)>" "\\1" IDIR_STRIPPED "${IDIR}")
        if(NOT IDIR_STRIPPED MATCHES "\\$<INSTALL_INTERFACE:" AND EXISTS "${IDIR_STRIPPED}")
            list(APPEND KEEL_CORE_INC_CLEAN "${IDIR_STRIPPED}")
        endif()
    endforeach()
    string(REPLACE ";" "|" KEEL_DEP_INC_JOINED "${KEEL_CORE_INC_CLEAN}")
    string(REPLACE ";" "|" KEEL_STD_INC_JOINED "${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES}")

    add_test(
        NAME test_header_purity
        COMMAND ${CMAKE_COMMAND}
            "-DKEEL_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/include"
            "-DKEEL_DEP_INC_DIRS=${KEEL_DEP_INC_JOINED}"
            "-DCOMPILER_PATH=${CMAKE_CXX_COMPILER}"
            "-DCOMPILER_ID=${CMAKE_CXX_COMPILER_ID}"
            "-DSTD_INC_DIRS=${KEEL_STD_INC_JOINED}"
            "-DCMAKE_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckHeaderPurity.cmake"
    )
endif()
