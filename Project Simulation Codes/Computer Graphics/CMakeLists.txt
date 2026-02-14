# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("Cherno")

# Tell cmake that we want to use C++ 20
set(CMAKE_CXX_STANDARD 20)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)        # Don't build Documentation
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)       # Don't build Tests
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)    # Don't build Examples
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)           # Don't build Installation Information
set(GLFW_USE_HYBRID_HPG ON CACHE BOOL "" FORCE)     # Add variables to use High Performance Graphics Card if available
# This tells cmake to include GLFW's cmake file into our cmake project.
# By doing so, the "glfw" library target will be defined, and we will be able to link it to our executable.
add_subdirectory(vendor/glfw EXCLUDE_FROM_ALL)

# This creates a new exectutable called "lab1" (e.g., on windows, it will be called "lab1.exe")
# The executable is created by compiling the following c/c++ files and linking so libraries (which we will define later)
add_executable(Cherno main.cpp vendor/glad/src/gl.c )
# This tells the compiler where to find the header files for GLFW and glad.
target_include_directories(Cherno PRIVATE 
   vendor/glfw/include
   vendor/glad/include
   vendor/glm
)

# This tells the linker to link the GLFW library to our executable.
target_link_libraries(Cherno PRIVATE glfw)
# Compilers on windows don't like the fopen function, so we disable the c-runtime security warnings
# This is like adding #define _CRT_SECURE_NO_WARNINGS at the start of each file
target_compile_definitions(Cherno PRIVATE _CRT_SECURE_NO_WARNINGS)

# Include sub-projects.
add_subdirectory ("Cherno")
