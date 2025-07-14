#!/bin/bash
# Test script for the enhanced extract_c_func.py with separate file extraction

echo "=== Enhanced C Function Extractor Test ==="
echo

# Clean up previous test files
rm -f test_*.c test_*.ll *_extracted.c

echo "1. Extract each function to separate files:"
echo "   Command: python3 extract_c_func.py test.c --separate"
python3 extract_c_func.py test.c --separate
echo

echo "2. Extract each function to separate files AND compile to LLVM IR:"
echo "   Command: python3 extract_c_func.py test.c --separate --compile-to-llvm"
python3 extract_c_func.py test.c --separate --compile-to-llvm
echo

echo "3. Extract to a specific output directory:"
mkdir -p extracted_functions
echo "   Command: python3 extract_c_func.py test.c --separate --output-dir extracted_functions"
python3 extract_c_func.py test.c --separate --output-dir extracted_functions
echo

echo "=== Generated Files ==="
echo "Main directory:"
ls -la test_*.c test_*.ll 2>/dev/null || echo "No files in main directory"
echo
echo "extracted_functions directory:"
ls -la extracted_functions/ 2>/dev/null || echo "No files in extracted_functions directory"
echo

echo "=== Sample Function File Content ==="
if [ -f "test_create_point.c" ]; then
    echo "Content of test_create_point.c:"
    cat test_create_point.c
    echo
fi

echo "=== Sample LLVM IR Content ==="
if [ -f "test_add_numbers.ll" ]; then
    echo "First 20 lines of test_add_numbers.ll:"
    head -20 test_add_numbers.ll
    echo
fi

echo "=== Usage Summary ==="
echo "Available options:"
echo "  --separate                    Extract each function to separate files"
echo "  --separate --compile-to-llvm  Also compile each file to LLVM IR"
echo "  --output-dir DIR              Specify output directory for separate files"
echo "  --all                         Extract all functions to one file"
echo "  -f FUNCTION_NAME              Extract specific function"
echo
echo "Files generated: $(ls test_*.c extracted_functions/*.c 2>/dev/null | wc -l)"
echo "LLVM IR files: $(ls test_*.ll 2>/dev/null | wc -l)"
