cmake_minimum_required(VERSION 3.15)
project(agentkv_core)

# 1. Find Python
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

# 2. Find nanobind - Use scikit-build-core's method
find_package(nanobind CONFIG REQUIRED HINTS ${Python_SITELIB})

# 3. Define the Extension Module
nanobind_add_module(agentkv_core 
    src/bindings.cpp 
    src/engine.cpp 
    src/slb.cpp
)

# 4. Compiler Flags for Performance
if(MSVC)
    target_compile_options(agentkv_core PRIVATE /O2 /std:c++20)
else()
    target_compile_options(agentkv_core PRIVATE -O3 -std=c++20 -fPIC -march=native)
endif()

# 5. Install the extension module at the wheel root (site-packages/)
# This avoids shadowing issues when running from the project directory.
install(TARGETS agentkv_core LIBRARY DESTINATION .)