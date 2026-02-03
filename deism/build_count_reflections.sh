#!/bin/bash
# Build script for count_reflections C++ library

# Determine the library extension based on platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    LIB_EXT=".dylib"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    LIB_EXT=".so"
else
    # Windows or other
    LIB_EXT=".so"
fi

echo "Compiling count_reflections.cpp..."
g++ -shared -fPIC -O3 -std=c++11 count_reflections.cpp -o count_reflections${LIB_EXT}

if [ $? -eq 0 ]; then
    echo "Successfully compiled count_reflections${LIB_EXT}"
    echo "You can now use count_reflections_wrapper.py to call the C++ function."
else
    echo "Compilation failed. Please check for errors."
    exit 1
fi


