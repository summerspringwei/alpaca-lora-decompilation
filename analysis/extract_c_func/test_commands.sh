#!/bin/bash
# Comprehensive test commands for extract_c_func.py

echo "=== C Function Extractor Usage Examples ==="
echo

# Clean up previous test files
rm -f *_extracted.c

echo "Usage: python3 extract_c_func.py [options] input_file"
echo "Options:"
echo "  -f, --function FUNC_NAME    Extract specific function"
echo "  --all                       Extract all functions"
echo "  -o, --output OUTPUT_FILE    Specify output file"
echo "  --clang-args ARGS           Additional clang arguments"
echo

echo "1. Extract a specific function with its dependencies:"
echo "   Command: python3 extract_c_func.py test.c -f create_point"
python3 extract_c_func.py test.c -f create_point
echo

echo "2. Extract a function that depends on other functions:"
echo "   Command: python3 extract_c_func.py test.c -f multiply_and_add"
python3 extract_c_func.py test.c -f multiply_and_add
echo

echo "3. Extract all functions to a specific output file:"
echo "   Command: python3 extract_c_func.py test.c --all -o my_extracted_functions.c"
python3 extract_c_func.py test.c --all -o my_extracted_functions.c
echo

echo "4. Extract with additional clang arguments (for complex projects):"
echo "   Command: python3 extract_c_func.py test.c -f print_color --clang-args -I/usr/include"
python3 extract_c_func.py test.c -f print_color --clang-args -I/usr/include
echo

echo "=== Generated Files ==="
ls -la *_extracted.c my_extracted_functions.c 2>/dev/null || echo "No files generated"
echo

echo "=== Sample Output (create_point function) ==="
if [ -f "test_extracted.c" ]; then
    echo "Content of test_extracted.c:"
    cat test_extracted.c
fi
