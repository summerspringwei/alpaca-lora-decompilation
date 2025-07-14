# Analysis Tools

This directory contains various analysis tools for the alpaca-lora-decompilation project.

## C Function Extractor

The C function extraction tools have been moved to the `extract_c_func/` subdirectory. This includes:
- `extract_c_func.py` - Main extraction script
- `test.c` - Sample C file for testing
- Test scripts and demonstrations
- Generated function files and LLVM IR files
- Documentation

To use the C function extractor:
```bash
cd extract_c_func
python3 extract_c_func.py --help
```

See `extract_c_func/README_extract_c_func.md` for detailed usage instructions.
