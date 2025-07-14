#!/usr/bin/env python3
"""
Extract C functions and their dependencies from C files using LLVM Python bindings.
This script uses clang's Python bindings to parse C code and extract function definitions
along with their dependencies (structs, typedefs, includes, etc.).
"""

import argparse
import os
import sys
from typing import Set, List, Dict, Optional
from pathlib import Path

try:
    import clang.cindex as clang
except ImportError:
    print("Error: clang Python bindings not found. Install with: pip install libclang")
    sys.exit(1)


class CFunctionExtractor:
    def __init__(self, source_file: str, clang_args: Optional[List[str]] = None):
        """
        Initialize the C function extractor.
        
        Args:
            source_file: Path to the C source file
            clang_args: Additional arguments to pass to clang parser
        """
        self.source_file = source_file
        self.clang_args = clang_args or []
        self.index = clang.Index.create()
        self.translation_unit = None
        self.extracted_functions = set()
        self.dependencies = {
            'includes': set(),
            'typedefs': set(),
            'structs': set(),
            'enums': set(),
            'macros': set(),
            'global_vars': set(),
            'function_declarations': set()
        }
        
    def parse_file(self):
        """Parse the C file using clang."""
        try:
            self.translation_unit = self.index.parse(
                self.source_file,
                args=self.clang_args,
                options=clang.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            )
            
            if not self.translation_unit:
                raise RuntimeError(f"Failed to parse {self.source_file}")
                
            # Check for parsing errors
            for diagnostic in self.translation_unit.diagnostics:
                if diagnostic.severity >= clang.Diagnostic.Error:
                    print(f"Parse error: {diagnostic.spelling}")
                    
        except Exception as e:
            raise RuntimeError(f"Error parsing file: {e}")
    
    def get_source_range_text(self, node) -> str:
        """Extract the source text for a given node."""
        try:
            with open(self.source_file, 'r') as f:
                content = f.read()
            
            start = node.extent.start
            end = node.extent.end
            
            lines = content.split('\n')
            
            if start.line == end.line:
                return lines[start.line - 1][start.column - 1:end.column - 1]
            else:
                result = [lines[start.line - 1][start.column - 1:]]
                for line_num in range(start.line, end.line - 1):
                    result.append(lines[line_num])
                result.append(lines[end.line - 1][:end.column - 1])
                return '\n'.join(result)
        except Exception:
            return ""
    
    def is_from_main_file(self, node) -> bool:
        """Check if a node is from the main source file (not from includes)."""
        if not node.location.file:
            return False
        return os.path.samefile(node.location.file.name, self.source_file)
    
    def extract_dependencies(self, node, visited: Optional[Set] = None):
        """Recursively extract dependencies for a given node."""
        if visited is None:
            visited = set()
            
        if node.hash in visited:
            return
        visited.add(node.hash)
        
        # Extract includes (preprocessing directives) - only from main file
        if node.kind == clang.CursorKind.INCLUSION_DIRECTIVE:
            if self.is_from_main_file(node):
                include_name = node.displayname
                # Ensure proper include format
                if not include_name.startswith('<') and not include_name.startswith('"'):
                    if include_name in ['stdio.h', 'stdlib.h', 'string.h', 'math.h', 'stddef.h']:
                        include_name = f"<{include_name}>"
                    else:
                        include_name = f'"{include_name}"'
                self.dependencies['includes'].add(f"#include {include_name}")
        
        # Extract typedefs
        elif node.kind == clang.CursorKind.TYPEDEF_DECL:
            if self.is_from_main_file(node):
                typedef_text = self.get_source_range_text(node)
                if typedef_text:
                    self.dependencies['typedefs'].add(typedef_text)
        
        # Extract struct definitions
        elif node.kind == clang.CursorKind.STRUCT_DECL:
            if self.is_from_main_file(node) and node.is_definition() and node.spelling:
                struct_text = self.get_source_range_text(node)
                if struct_text:
                    self.dependencies['structs'].add(struct_text)
        
        # Extract enum definitions
        elif node.kind == clang.CursorKind.ENUM_DECL:
            if self.is_from_main_file(node) and node.is_definition():
                enum_text = self.get_source_range_text(node)
                if enum_text:
                    self.dependencies['enums'].add(enum_text)
        
        # Extract global variable declarations
        elif node.kind == clang.CursorKind.VAR_DECL:
            if self.is_from_main_file(node) and node.linkage == clang.LinkageKind.EXTERNAL:
                var_text = self.get_source_range_text(node)
                if var_text:
                    self.dependencies['global_vars'].add(var_text)
        
        # Extract function declarations (not definitions)
        elif node.kind == clang.CursorKind.FUNCTION_DECL:
            if self.is_from_main_file(node) and not node.is_definition():
                func_text = self.get_source_range_text(node)
                if func_text:
                    self.dependencies['function_declarations'].add(func_text)
        
        # Recursively process children
        for child in node.get_children():
            self.extract_dependencies(child, visited)
    
    def find_function_dependencies(self, function_node):
        """Find all dependencies for a specific function."""
        # This is a simplified dependency analysis
        # In a full implementation, you'd want to do more sophisticated analysis
        # of what types, functions, and variables are actually used within the function
        
        def visit_function_body(node):
            if node.kind == clang.CursorKind.DECL_REF_EXPR:
                # Reference to a declaration - could be a function call or variable reference
                referenced = node.referenced
                if referenced and referenced.kind == clang.CursorKind.FUNCTION_DECL:
                    if self.is_from_main_file(referenced):
                        func_text = self.get_source_range_text(referenced)
                        if func_text and referenced.is_definition():
                            self.extracted_functions.add(func_text)
                        elif func_text and not referenced.is_definition():
                            self.dependencies['function_declarations'].add(func_text)
            
            for child in node.get_children():
                visit_function_body(child)
        
        visit_function_body(function_node)
    
    def extract_function(self, function_name: str) -> bool:
        """
        Extract a specific function and its dependencies.
        
        Args:
            function_name: Name of the function to extract
            
        Returns:
            True if function was found and extracted, False otherwise
        """
        if not self.translation_unit:
            self.parse_file()
        
        # First pass: collect all dependencies from the entire file
        self.extract_dependencies(self.translation_unit.cursor)
        
        # Second pass: find the specific function
        def find_function(node):
            if (node.kind == clang.CursorKind.FUNCTION_DECL and 
                node.spelling == function_name and 
                node.is_definition() and 
                self.is_from_main_file(node)):
                
                func_text = self.get_source_range_text(node)
                if func_text:
                    self.extracted_functions.add(func_text)
                    # Find dependencies specific to this function
                    self.find_function_dependencies(node)
                    return True
            
            for child in node.get_children():
                if find_function(child):
                    return True
            return False
        
        return find_function(self.translation_unit.cursor)
    
    def extract_all_functions(self):
        """Extract all functions from the file."""
        if not self.translation_unit:
            self.parse_file()
        
        # Collect all dependencies
        self.extract_dependencies(self.translation_unit.cursor)
        
        # Find all function definitions
        def find_all_functions(node):
            if (node.kind == clang.CursorKind.FUNCTION_DECL and 
                node.is_definition() and 
                self.is_from_main_file(node)):
                
                func_text = self.get_source_range_text(node)
                if func_text:
                    self.extracted_functions.add(func_text)
                    self.find_function_dependencies(node)
            
            for child in node.get_children():
                find_all_functions(child)
        
        find_all_functions(self.translation_unit.cursor)
    
    def save_extracted_code(self, output_file: str, include_header: bool = True):
        """
        Save the extracted functions and dependencies to a new file.
        
        Args:
            output_file: Path to the output file
            include_header: Whether to include a header comment
        """
        with open(output_file, 'w') as f:
            if include_header:
                f.write(f"/* Extracted from {self.source_file} */\n\n")
            
            # Write includes
            if self.dependencies['includes']:
                f.write("/* Includes */\n")
                for include in sorted(self.dependencies['includes']):
                    f.write(f"{include}\n")
                f.write("\n")
            
            # Write typedefs
            if self.dependencies['typedefs']:
                f.write("/* Typedefs */\n")
                for typedef in sorted(self.dependencies['typedefs']):
                    f.write(f"{typedef};\n")
                f.write("\n")
            
            # Write enums
            if self.dependencies['enums']:
                f.write("/* Enums */\n")
                for enum in sorted(self.dependencies['enums']):
                    f.write(f"{enum};\n\n")
            
            # Write structs
            if self.dependencies['structs']:
                f.write("/* Structs */\n")
                for struct in sorted(self.dependencies['structs']):
                    f.write(f"{struct};\n\n")
            
            # Write global variables
            if self.dependencies['global_vars']:
                f.write("/* Global Variables */\n")
                for var in sorted(self.dependencies['global_vars']):
                    f.write(f"{var};\n")
                f.write("\n")
            
            # Write function declarations
            if self.dependencies['function_declarations']:
                f.write("/* Function Declarations */\n")
                for func_decl in sorted(self.dependencies['function_declarations']):
                    f.write(f"{func_decl};\n")
                f.write("\n")
            
            # Write function definitions
            if self.extracted_functions:
                f.write("/* Function Definitions */\n")
                for func in sorted(self.extracted_functions):
                    f.write(f"{func}\n\n")

    def save_function_with_dependencies(self, function_text: str, function_name: str, output_file: str):
        """
        Save a single function with all its dependencies to a separate file.
        
        Args:
            function_text: The function's source code text
            function_name: Name of the function
            output_file: Path to the output file
        """
        with open(output_file, 'w') as f:
            f.write(f"/* Function: {function_name} - Extracted from {self.source_file} */\n\n")
            
            # Write includes
            if self.dependencies['includes']:
                for include in sorted(self.dependencies['includes']):
                    f.write(f"{include}\n")
                f.write("\n")
            
            # Write typedefs
            if self.dependencies['typedefs']:
                for typedef in sorted(self.dependencies['typedefs']):
                    f.write(f"{typedef};\n")
                f.write("\n")
            
            # Write enums
            if self.dependencies['enums']:
                for enum in sorted(self.dependencies['enums']):
                    f.write(f"{enum};\n\n")
            
            # Write structs
            if self.dependencies['structs']:
                for struct in sorted(self.dependencies['structs']):
                    f.write(f"{struct};\n\n")
            
            # Write global variables
            if self.dependencies['global_vars']:
                for var in sorted(self.dependencies['global_vars']):
                    f.write(f"{var};\n")
                f.write("\n")
            
            # Write function declarations
            if self.dependencies['function_declarations']:
                for func_decl in sorted(self.dependencies['function_declarations']):
                    f.write(f"{func_decl};\n")
                f.write("\n")
            
            # Write the specific function
            f.write(f"{function_text}\n")

    def extract_all_functions_separately(self, output_dir: str = None) -> List[str]:
        """
        Extract all functions from the file and save each to a separate file.
        
        Args:
            output_dir: Directory to save individual function files (default: same as source)
            
        Returns:
            List of generated file paths
        """
        if not self.translation_unit:
            self.parse_file()
        
        # Collect all dependencies first
        self.extract_dependencies(self.translation_unit.cursor)
        
        # Set output directory
        if output_dir is None:
            output_dir = os.path.dirname(self.source_file)
        
        generated_files = []
        
        # Find all function definitions
        def find_all_functions(node):
            if (node.kind == clang.CursorKind.FUNCTION_DECL and 
                node.is_definition() and 
                self.is_from_main_file(node)):
                
                func_text = self.get_source_range_text(node)
                func_name = node.spelling
                
                if func_text and func_name:
                    # Create output filename
                    base_name = os.path.splitext(os.path.basename(self.source_file))[0]
                    output_file = os.path.join(output_dir, f"{base_name}_{func_name}.c")
                    
                    # Save function with dependencies
                    self.save_function_with_dependencies(func_text, func_name, output_file)
                    generated_files.append(output_file)
                    
                    print(f"Extracted function '{func_name}' to {output_file}")
            
            for child in node.get_children():
                find_all_functions(child)
        
        find_all_functions(self.translation_unit.cursor)
        return generated_files


def main():
    parser = argparse.ArgumentParser(
        description="Extract C functions and dependencies using LLVM Python bindings"
    )
    parser.add_argument("input_file", help="Input C source file")
    parser.add_argument("-o", "--output", help="Output file (default: <input>_extracted.c)")
    parser.add_argument("-f", "--function", help="Specific function name to extract")
    parser.add_argument("--all", action="store_true", help="Extract all functions to one file")
    parser.add_argument("--separate", action="store_true", help="Extract each function to separate files")
    parser.add_argument("--output-dir", help="Output directory for separate files (default: same as input)")
    parser.add_argument("--clang-args", nargs="*", help="Additional arguments for clang parser")
    parser.add_argument("--compile-to-llvm", action="store_true", help="Also compile extracted files to LLVM IR")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Check for conflicting options
    if sum([bool(args.function), args.all, args.separate]) != 1:
        print("Error: Must specify exactly one of --function <name>, --all, or --separate")
        sys.exit(1)
    
    # Default output filename for single file extraction
    if not args.output and not args.separate:
        input_path = Path(args.input_file)
        args.output = str(input_path.parent / f"{input_path.stem}_extracted.c")
    
    try:
        extractor = CFunctionExtractor(args.input_file, args.clang_args)
        
        if args.function:
            print(f"Extracting function '{args.function}' from {args.input_file}")
            if extractor.extract_function(args.function):
                print(f"Function '{args.function}' found and extracted")
                extractor.save_extracted_code(args.output)
                print(f"Extracted code saved to {args.output}")
            else:
                print(f"Function '{args.function}' not found")
                sys.exit(1)
                
        elif args.all:
            print(f"Extracting all functions from {args.input_file}")
            extractor.extract_all_functions()
            extractor.save_extracted_code(args.output)
            print(f"Extracted code saved to {args.output}")
            
        elif args.separate:
            print(f"Extracting each function separately from {args.input_file}")
            generated_files = extractor.extract_all_functions_separately(args.output_dir)
            
            if generated_files:
                print(f"\nGenerated {len(generated_files)} separate function files:")
                for file_path in generated_files:
                    print(f"  - {file_path}")
                
                # Optionally compile to LLVM IR
                if args.compile_to_llvm:
                    print("\nCompiling to LLVM IR...")
                    compile_to_llvm_ir(generated_files)
            else:
                print("No functions found to extract")
        
        # Print summary
        print(f"\nSummary:")
        if args.separate:
            print(f"  Files generated: {len(generated_files) if 'generated_files' in locals() else 0}")
        else:
            print(f"  Functions extracted: {len(extractor.extracted_functions)}")
        print(f"  Includes: {len(extractor.dependencies['includes'])}")
        print(f"  Typedefs: {len(extractor.dependencies['typedefs'])}")
        print(f"  Structs: {len(extractor.dependencies['structs'])}")
        print(f"  Enums: {len(extractor.dependencies['enums'])}")
        print(f"  Global variables: {len(extractor.dependencies['global_vars'])}")
        print(f"  Function declarations: {len(extractor.dependencies['function_declarations'])}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def compile_to_llvm_ir(c_files: List[str]):
    """
    Compile C files to LLVM IR using clang.
    
    Args:
        c_files: List of C file paths to compile
    """
    import subprocess
    
    for c_file in c_files:
        ll_file = c_file.replace('.c', '.ll')
        try:
            # Compile to LLVM IR
            cmd = ['clang', '-S', '-emit-llvm', '-O0', c_file, '-o', ll_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  ✓ {c_file} -> {ll_file}")
            else:
                print(f"  ✗ Failed to compile {c_file}: {result.stderr.strip()}")
                
        except FileNotFoundError:
            print(f"  ✗ clang not found. Please install LLVM/Clang to compile to IR")
            break
        except Exception as e:
            print(f"  ✗ Error compiling {c_file}: {e}")


if __name__ == "__main__":
    main()