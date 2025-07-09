
## How analyze real-world large scale project


### How to compile each .cc file to .ll
The core idea is how to capture the building commands of `make`,
we use a tool [bear](https://github.com/rizsotto/Bear) to generates a compilation database for clang tooling.
Here is a example:
```shell
bear make CC=/home/xiachunwei/Software//llvm-project-release/myrelease-17.0.2/bin/clang CFLAGS="-O2 -Wno-error=unused-command-line-argument" MALLOC=libc
```
This will generate a `compile_commands.json` that record all the commands and arguments during building.

Then we change the compiling commands, convert compiling to object file `.o` to LLVM IR file `.ll`.
execute the following:
```shell
python3 generate_ll_from_compile_commands.py
```
