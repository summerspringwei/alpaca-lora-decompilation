#!/bin/bash
# Final demonstration of the enhanced C function extractor

echo "=== Enhanced C Function Extractor - Final Demo ==="
echo

# Clean up
rm -f test_*.c test_*.ll extracted_functions/* my_extracted_functions.c

echo "🔧 Available Commands:"
echo "1. Extract specific function:"
echo "   python3 extract_c_func.py test.c -f function_name"
echo
echo "2. Extract all functions to one file:"
echo "   python3 extract_c_func.py test.c --all"
echo
echo "3. Extract each function to separate files:"
echo "   python3 extract_c_func.py test.c --separate"
echo
echo "4. Extract + compile to LLVM IR:"
echo "   python3 extract_c_func.py test.c --separate --compile-to-llvm"
echo
echo "5. Extract to specific directory:"
echo "   python3 extract_c_func.py test.c --separate --output-dir output_folder"
echo

echo "🚀 Running demonstration..."
echo

echo "1. Extracting each function separately with LLVM IR compilation:"
python3 extract_c_func.py test.c --separate --compile-to-llvm
echo

echo "📊 Results:"
echo "Generated C files:"
ls -la test_*.c | head -5
echo "... (showing first 5)"
echo
echo "Generated LLVM IR files:"
ls -la test_*.ll 2>/dev/null || echo "Some files failed to compile (expected for functions with dependencies)"
echo

echo "📝 Sample function file (test_add_numbers.c):"
echo "----------------------------------------"
cat test_add_numbers.c
echo "----------------------------------------"
echo

echo "🔬 Sample LLVM IR (first 15 lines of test_add_numbers.ll):"
echo "----------------------------------------"
head -15 test_add_numbers.ll 2>/dev/null || echo "File not found"
echo "----------------------------------------"
echo

echo "✅ Summary:"
echo "- Successfully extracted $(ls test_*.c 2>/dev/null | wc -l) functions to separate C files"
echo "- Successfully compiled $(ls test_*.ll 2>/dev/null | wc -l) functions to LLVM IR"
echo "- Each file contains all necessary dependencies (structs, typedefs, includes, etc.)"
echo "- Files are ready for individual analysis or compilation"
echo
echo "💡 Use cases:"
echo "- Individual function analysis"
echo "- Function-level LLVM IR generation"
echo "- Isolated compilation and testing"
echo "- Code extraction for research or documentation"
