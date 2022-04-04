
# Benchmark

Rewrite this README to explain what the benchmark is!

## About the template

Copy this directory into `benchmarks/mybench` and adapt the contents. Not all files may be necessary.

The main file to look at is `benchfile.py`. It defines what to do on install/prepare/run, but by default these phases will respectively use/execute the requirements.txt/prepare.py/main.py files.

Try out the template:

```bash
milabench install benchmarks/mybench/dev.yaml
milabench prepare benchmarks/mybench/dev.yaml --dev
milabench run benchmarks/mybench/dev.yaml --dev
```

The `--dev` flag will sync any of your changes to the copy of the code in `$MILABENCH_BASE` and it will force the use of only one job. (There is also `--sync` to only do the sync).
