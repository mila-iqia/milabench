
# Benchmark

Rewrite this README to explain what the benchmark is!

## About the template

Copy this directory into `benchmarks/mybench` and adapt the contents. Not all files may be necessary.

The main file to look at is `benchfile.py`. It defines what to do on install/prepare/run, but by default these phases will respectively use/execute the requirements.txt/prepare.py/main.py files.

Try out the template:

```bash
milabench install benchmarks/mybench/dev.yaml
milabench prepare benchmarks/mybench/dev.yaml
milabench run benchmarks/mybench/dev.yaml
```

If you change anything in the benchmark directory and you want to test, you can sync the changes to the copy of the code in `$MILABENCH_BASE` with:

```bash
milabench install benchmarks/mybench/dev.yaml --sync
```
