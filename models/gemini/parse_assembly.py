"""
This script extracts all the called function names from an x86 assembly file.
"""

import re

def extract_called_functions(assembly_file_path):
    """
    Reads an x86 assembly file and extracts all the called function names.

    Args:
        assembly_file_path (str): Path to the assembly file.

    Returns:
        list: A list of unique function names called in the assembly file.
    """
    called_functions = set()
    call_instruction_pattern = re.compile(r'\bcall\s+([a-zA-Z_][a-zA-Z0-9_]*)')

    with open(assembly_file_path, 'r') as file:
        for line in file:
            match = call_instruction_pattern.search(line)
            if match:
                called_functions.add(match.group(1))

    return sorted(called_functions)

# Example usage:
# functions = extract_called_functions('example.asm')
# print(functions)