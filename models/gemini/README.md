
## Coreutils study
Complie `basename` to llvm IR and assembly:
```bash
FLAGS="-Werror -Wall"
clang  -I. -I./lib  -Ilib -I./lib -Isrc -I./src $FLAGS -O2 -MT src/basename.o -MD -MP -MF $depbase.Tpo -S -fno-asynchronous-unwind-tables -emit-llvm -o src/basename.ll src/basename.c
clang  -I. -I./lib  -Ilib -I./lib -Isrc -I./src $FLAGS -O2 -MT src/basename.o -MD -MP -MF $depbase.Tpo -S -fno-asynchronous-unwind-tables -o src/basename.s src/basename.c
```
Note the option `-fno-asynchronous-unwind-tables` will strip the debug info like ".cfi_xxx"


### The overall workflow:

1. Extract one function from LLVM IR file:
```bash
llvm-extract --func=main -S src/tail.ll -o src/tail_main.ll
```

2. Delete one function from module:
```bash
llvm-extract -delete -func=main -S src/tail.ll -o src/tail_no_main.ll
```

3. Compile function LLVM IR to assembly:
```bash
llc src/tail_main.ll -o src/tail_main.s
```

4. Use the LLM to decompile assembly to IR


5. link the decompiled `.ll` to one `.ll` file:
```bash
llvm-link src/tail_no_main.ll src/tail_main.ll -o tail.ll
```

6. Compile LLVM IR to object file:
```bash
clang -c src/tail.ll -o tail.o
```

7. link the object file to the executable binary:
```bash
clang  -Wno-format-extra-args -Wno-implicit-const-int-float-conversion -Wno-tautological-constant-out-of-range-compare -g -O2 -Wl,--as-needed  -o src/tail src/tail.o src/iopoll.o src/libver.a lib/libcoreutils.a   lib/libcoreutils.a  -ldl
```

8. Run the test:
```bash
make TESTS=tests/tail/overlay-headers.sh check VERBOSE=yes
```
