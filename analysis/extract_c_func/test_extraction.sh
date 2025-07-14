#!/bin/bash
# Test script for extract_c_func.py

echo "=== Testing C Function Extractor ==="
echo

# Check if libclang is installed
python3 -c "import clang.cindex" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing libclang..."
    pip install libclang
    echo
fi

echo "1. Extract a specific function (create_point):"
python3 extract_c_func.py test.c -f create_point -o test_create_point_extracted.c
echo

echo "2. Extract another function (multiply_and_add) with its dependencies:"
python3 extract_c_func.py test.c -f multiply_and_add -o test_multiply_extracted.c
echo

echo "3. Extract all functions:"
python3 extract_c_func.py test.c --all -o test_all_extracted.c
echo

echo "4. Show extracted files:"
echo "--- test_create_point_extracted.c ---"
head -20 test_create_point_extracted.c
echo
echo "--- test_multiply_extracted.c ---"
head -20 test_multiply_extracted.c
echo
echo "--- test_all_extracted.c (first 30 lines) ---"
head -30 test_all_extracted.c
echo

echo "=== Test completed ==="
echo "Generated files:"
ls -la *_extracted.c 2>/dev/null || echo "No extracted files found"
