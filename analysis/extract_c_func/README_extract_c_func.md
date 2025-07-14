# C Function Extractor with LLVM Python Bindings

This tool extracts C functions and their dependencies from C source files using LLVM's clang Python bindings. It can extract individual functions or all functions to separate files, optionally compiling them to LLVM IR.

## Features

- Extract individual functions with all their dependencies
- Extract all functions to separate files
- Automatically include necessary headers, typedefs, structs, enums, and global variables
- Compile extracted functions to LLVM IR
- Support for custom output directories

## Installation

```bash
pip install libclang
```

## Usage

### Basic Commands

1. **Extract a specific function:**
```bash
python3 extract_c_func.py input.c -f function_name
```

2. **Extract all functions to one file:**
```bash
python3 extract_c_func.py input.c --all
```

3. **Extract each function to separate files:**
```bash
python3 extract_c_func.py input.c --separate
```

4. **Extract + compile to LLVM IR:**
```bash
python3 extract_c_func.py input.c --separate --compile-to-llvm
```

5. **Extract to specific directory:**
```bash
python3 extract_c_func.py input.c --separate --output-dir output_folder
```

### Command Line Options

- `-f, --function FUNC_NAME`: Extract specific function
- `--all`: Extract all functions to one file
- `--separate`: Extract each function to separate files
- `-o, --output FILE`: Specify output file (for single file extraction)
- `--output-dir DIR`: Specify output directory (for separate files)
- `--compile-to-llvm`: Also compile extracted files to LLVM IR
- `--clang-args ARGS`: Additional arguments for clang parser

## Example Output

For a function `add_numbers`, the extracted file `test_add_numbers.c` will contain:

```c
/* Function: add_numbers - Extracted from test.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int x;
    int y;
} Point;

enum Color {
    RED,
    GREEN,
    BLUE
};

int global_counter = 0;

int helper_function(int a, int b);

int add_numbers(int a, int b) {
    return a + b;
}
```

## LLVM IR Compilation

When using `--compile-to-llvm`, the tool will automatically compile each extracted C file to LLVM IR using clang:

```bash
clang -S -emit-llvm -O0 extracted_function.c -o extracted_function.ll
```

## Use Cases

- **Individual function analysis**: Extract specific functions for detailed analysis
- **Function-level LLVM IR generation**: Generate LLVM IR for individual functions
- **Isolated compilation and testing**: Test functions in isolation
- **Code extraction for research**: Extract functions with all dependencies for research purposes
- **Documentation**: Create standalone examples of functions

## Test Files

The repository includes test files:
- `test.c`: Sample C file with various function types
- `test_separate_extraction.sh`: Test script demonstrating all features
- `demo_final.sh`: Comprehensive demonstration

## Notes

- Functions that call other functions from the same file may not compile individually
- System headers are automatically detected and properly formatted
- Anonymous structs may generate warnings but don't prevent compilation
- The tool preserves all necessary dependencies for each extracted function
