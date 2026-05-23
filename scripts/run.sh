

export MILABENCH_GPU_ARCH=rocm
export MILABENCH_CONFIG=/home/amd/milabench/config/all.yaml
export MILABENCH_VENV=/home/amd/milabench/.venv

export MILABENCH_WORDIR="/data/output/$MILABENCH_GPU_ARCH"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"
export MILABENCH_SIZER_SAVE="$MILABENCH_WORDIR/scaling.yaml"

cd /home/amd/milabench/
mkdir -p $MILABENCH_BASE

uv venv --python=3.12
(
        . $MILABENCH_VENV/bin/activate

        # shouldn't be necessary anymore, unless you modify the milabench/constraints/rocm.txt
        # milabench pin --variant rocm --from-scratch

        milabench install       # --select $BENCHMARK
        (
                . $BENCHMARK_VENV/bin/activate
                
                # Override the dependencies used by the benchmarks
                # uv pip install ...

        )
        milabench prepare       # --select $BENCHMARK

        export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

        milabench run           # --select $BENCHMARK
)





# uv pip install --extra-index-url https://wheels.vllm.ai/rocm/ vllm


# 
# https://rocm.frameworks.amd.com/whl/gfx950-dcgpu/

#  PYTHON 3.13

source .venv/bin/activate
uv pip install --extra-index-url https://repo.amd.com/rocm/whl/gfx950-dcgpu/  torch==2.11.0+rocm7.13.0 torchaudio==2.11.0+rocm7.13.0 torchvision==0.26.0+rocm7.13.0
uv pip install https://rocm.frameworks.amd.com/whl/gfx950-dcgpu/amd_aiter-0.1.10.post2-cp313-cp313-linux_x86_64.whl
uv pip install https://rocm.frameworks.amd.com/whl/gfx950-dcgpu/flash_attn-2.8.3-cp313-cp313-linux_x86_64.whl

uv pip install https://rocm.frameworks.amd.com/whl/gfx950-dcgpu/vllm-0.19.1.dev3%2Brocm7.13.0.g24efb8904.d20260514-cp313-cp313-linux_x86_64.whl
# uv pip install --extra-index-url https://rocm.frameworks.amd.com/whl/gfx950-dcgpu/ vllm



# uv pip install --extra-index-url https://rocm.frameworks.amd.com/whl/gfx950-dcgpu/ vllm
#   × No solution found when resolving dependencies:
#   ╰─▶ Because only the following versions of vllm are available:
#           vllm==0.16.1.dev10+g11515110f.d20260324.rocm712
#           vllm==0.19.1.dev3+rocm7.13.0.g24efb8904.d20260514
#       and vllm==0.16.1.dev10+g11515110f.d20260324.rocm712 has no wheels with a matching Python ABI tag
#       (e.g., `cp313`), we can conclude that vllm<0.19.1.dev3+rocm7.13.0.g24efb8904.d20260514 cannot be
#       used. (1)

#       Because amd-aiter was not found in the package registry and
#       vllm==0.19.1.dev3+rocm7.13.0.g24efb8904.d20260514 depends on amd-aiter==0.1.10.post2, we can
#       conclude that vllm==0.19.1.dev3+rocm7.13.0.g24efb8904.d20260514 cannot be used.
#       And because we know from (1) that vllm<0.19.1.dev3+rocm7.13.0.g24efb8904.d20260514 cannot be used,
#       we can conclude that all versions of vllm cannot be used.
#       And because you require vllm, we can conclude that your requirements are unsatisfiable.

#       hint: `vllm` was requested with a pre-release marker (e.g., all of:
#           vllm<0.16.1.dev10+g11515110f.d20260324.rocm712
#           vllm>0.16.1.dev10+g11515110f.d20260324.rocm712,<0.19.1.dev3+rocm7.13.0.g24efb8904.d20260514
#           vllm>0.19.1.dev3+rocm7.13.0.g24efb8904.d20260514
#       ), but pre-releases weren't enabled (try: `--prerelease=allow`)

#       hint: `vllm` was found on https://rocm.frameworks.amd.com/whl/gfx950-dcgpu/, but not at the
#       requested version (all of:
#           vllm<0.16.1.dev10+g11515110f.d20260324.rocm712
#           vllm>0.16.1.dev10+g11515110f.d20260324.rocm712,<0.19.1.dev3+rocm7.13.0.g24efb8904.d20260514
#           vllm>0.19.1.dev3+rocm7.13.0.g24efb8904.d20260514
#       ). A compatible version may be available on a subsequent index (e.g., https://pypi.org/simple).
#       By default, uv will only consider versions that are published on the first index that contains
#       a given package, to avoid dependency confusion attacks. If all indexes are equally trusted, use
#       `--index-strategy unsafe-best-match` to consider all versions from all indexes, regardless of the
#       order in which they were defined.

#       hint: An index URL (https://rocm.frameworks.amd.com/whl/gfx950-dcgpu/) returned a 403 Forbidden
#       error. This could indicate lack of valid authentication credentials, or the package may not exist on
#       this index.

#       hint: You require CPython 3.13 (`cp313`), but we only found wheels for `vllm`
#       (v0.16.1.dev10+g11515110f.d20260324.rocm712) with the following Python ABI tag: `cp312`


# Resolved 209 packages in 208ms
# Prepared 54 packages in 2.41s
# Uninstalled 2 packages in 21ms
# error: Failed to install: vllm-0.19.1.dev3+rocm7.13.0.g24efb8904.d20260514-cp313-cp313-linux_x86_64.whl (vllm==0.19.1.dev3+rocm7.13.0.g24efb8904.d20260514 (from https://rocm.frameworks.amd.com/whl/gfx950-dcgpu/vllm-0.19.1.dev3%2Brocm7.13.0.g24efb8904.d20260514-cp313-cp313-linux_x86_64.whl))
#   Caused by: Wheel version does not match filename (0.19.2.dev3+rocm7.13.0rc2.g24efb8904.d20260514 != 0.19.1.dev3+rocm7.13.0.g24efb8904.d20260514), which indicates a malformed wheel. If this is intentional, set `UV_SKIP_WHEEL_FILENAME_CHECK=1`.







{"event": "line", "data": "WARNING 05-15 13:20:45 [interface.py:229] Failed to import from vllm._C: ImportError(\"/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found (required by /home/amd/.venv/lib/python3.13/site-packages/vllm/_C.abi3.so)\")\n", "pipe": "stdout"}
{"event": "line", "data": "We recommend installing via `pip install torch-c-dlpack-ext`\n", "pipe": "stderr"}
{"event": "line", "data": "  warnings.warn(\n", "pipe": "stderr"}
{"event": "line", "data": "Traceback (most recent call last):\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"/home/amd/.venv/bin/vllm\", line 10, in <module>\n", "pipe": "stderr"}
{"event": "line", "data": "    sys.exit(main())\n", "pipe": "stderr"}
{"event": "line", "data": "             ~~~~^^\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"/home/amd/.venv/lib/python3.13/site-packages/vllm/entrypoints/cli/main.py\", line 68, in main\n", "pipe": "stderr"}
{"event": "line", "data": "    cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)\n", "pipe": "stderr"}
{"event": "line", "data": "    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"/home/amd/.venv/lib/python3.13/site-packages/vllm/entrypoints/cli/serve.py\", line 138, in subparser_init\n", "pipe": "stderr"}
{"event": "line", "data": "    serve_parser = make_arg_parser(serve_parser)\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"/home/amd/.venv/lib/python3.13/site-packages/vllm/entrypoints/openai/cli_args.py\", line 349, in make_arg_parser\n", "pipe": "stderr"}
{"event": "line", "data": "    parser = AsyncEngineArgs.add_cli_args(parser)\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"/home/amd/.venv/lib/python3.13/site-packages/vllm/engine/arg_utils.py\", line 2264, in add_cli_args\n", "pipe": "stderr"}
{"event": "line", "data": "    parser = EngineArgs.add_cli_args(parser)\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"/home/amd/.venv/lib/python3.13/site-packages/vllm/engine/arg_utils.py\", line 1294, in add_cli_args\n", "pipe": "stderr"}
{"event": "line", "data": "    vllm_kwargs = get_kwargs(VllmConfig)\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"/home/amd/.venv/lib/python3.13/site-packages/vllm/engine/arg_utils.py\", line 369, in get_kwargs\n", "pipe": "stderr"}
{"event": "line", "data": "    return copy.deepcopy(_compute_kwargs(cls))\n", "pipe": "stderr"}
{"event": "line", "data": "                         ~~~~~~~~~~~~~~~^^^^^\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"/home/amd/.venv/lib/python3.13/site-packages/vllm/engine/arg_utils.py\", line 280, in _compute_kwargs\n", "pipe": "stderr"}
{"event": "line", "data": "    default = default.default_factory()  # type: ignore[call-arg]\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"/home/amd/.venv/lib/python3.13/site-packages/pydantic/_internal/_dataclasses.py\", line 121, in __init__\n", "pipe": "stderr"}
{"event": "line", "data": "    s.__pydantic_validator__.validate_python(ArgsKwargs(args, kwargs), self_instance=s)\n", "pipe": "stderr"}
{"event": "line", "data": "    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"/home/amd/.venv/lib/python3.13/site-packages/vllm/config/device.py\", line 56, in __post_init__\n", "pipe": "stderr"}
{"event": "line", "data": "    raise RuntimeError(\n", "pipe": "stderr"}
{"event": "line", "data": "    ...<3 lines>...\n", "pipe": "stderr"}
{"event": "line", "data": "    )\n", "pipe": "stderr"}
{"event": "line", "data": "RuntimeError: Failed to infer device type, please set the environment variable `VLLM_LOGGING_LEVEL=DEBUG` to turn on verbose logging to help debug the issue.\n", "pipe": "stderr"}



Have to uninstall flashinfer-python


{"event": "line", "data": "!!!!!!! Segfault encountered !!!!!!!\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in void ck_tile::launch_and_check<ck_tile::make_kernel<256, 3, ck_tile::FmhaFwdKernel<ck_tile::BlockFmhaPipelineQRKSVSAsync<ck_tile::BlockFmhaPipelineProblem<unsigned short, unsigned short, unsigned short, float, float, unsigned short, unsigned short, float, unsigned short, float, unsigned short, ck_tile::TileFmhaShape<ck_tile::sequence<128, 64, 32, 64, 32, 64>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, true>, true, ck_tile::ComposedAttention<0u, true>, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::TileFmhaTraits<true, true, true, true, false, (ck_tile::BlockAttentionBiasEnum)0, false, false, false, false, -1, false> >, ck_tile::BlockFmhaPipelineQXKSVSCustomPolicy<true, true, 3, 3> >, ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, unsigned short, true, true, true, (ck_tile::memory_operation_enum)0>, void> >, ck_tile::FmhaFwdKernel<ck_tile::BlockFmhaPipelineQRKSVSAsync<ck_tile::BlockFmhaPipelineProblem<unsigned short, unsigned short, unsigned short, float, float, unsigned short, unsigned short, float, unsigned short, float, unsigned short, ck_tile::TileFmhaShape<ck_tile::sequence<128, 64, 32, 64, 32, 64>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, true>, true, ck_tile::ComposedAttention<0u, true>, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::TileFmhaTraits<true, true, true, true, false, (ck_tile::BlockAttentionBiasEnum)0, false, false, false, false, -1, false> >, ck_tile::BlockFmhaPipelineQXKSVSCustomPolicy<true, true, 3, 3> >, ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, unsigned short, true, true, true, (ck_tile::memory_operation_enum)0>, void> >::FmhaFwdGroupModeKargs>(ck_tile::FmhaFwdKernel<ck_tile::BlockFmhaPipelineQRKSVSAsync<ck_tile::BlockFmhaPipelineProblem<unsigned short, unsigned short, unsigned short, float, float, unsigned short, unsigned short, float, unsigned short, float, unsigned short, ck_tile::TileFmhaShape<ck_tile::sequence<128, 64, 32, 64, 32, 64>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, true>, true, ck_tile::ComposedAttention<0u, true>, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::TileFmhaTraits<true, true, true, true, false, (ck_tile::BlockAttentionBiasEnum)0, false, false, false, false, -1, false> >, ck_tile::BlockFmhaPipelineQXKSVSCustomPolicy<true, true, 3, 3> >, ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, unsigned short, true, true, true, (ck_tile::memory_operation_enum)0>, void> >, dim3, dim3, unsigned long, ck_tile::FmhaFwdKernel<ck_tile::BlockFmhaPipelineQRKSVSAsync<ck_tile::BlockFmhaPipelineProblem<unsigned short, unsigned short, unsigned short, float, float, unsigned short, unsigned short, float, unsigned short, float, unsigned short, ck_tile::TileFmhaShape<ck_tile::sequence<128, 64, 32, 64, 32, 64>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, true>, true, ck_tile::ComposedAttention<0u, true>, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::TileFmhaTraits<true, true, true, true, false, (ck_tile::BlockAttentionBiasEnum)0, false, false, false, false, -1, false> >, ck_tile::BlockFmhaPipelineQXKSVSCustomPolicy<true, true, 3, 3> >, ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, unsigned short, true, true, true, (ck_tile::memory_operation_enum)0>, void> >::FmhaFwdGroupModeKargs)::{lambda(ck_tile::stream_config const&)#1}>(ck_tile::stream_config const&, ck_tile::make_kernel<256, 3, ck_tile::FmhaFwdKernel<ck_tile::BlockFmhaPipelineQRKSVSAsync<ck_tile::BlockFmhaPipelineProblem<unsigned short, unsigned short, unsigned short, float, float, unsigned short, unsigned short, float, unsigned short, float, unsigned short, ck_tile::TileFmhaShape<ck_tile::sequence<128, 64, 32, 64, 32, 64>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, true>, true, ck_tile::ComposedAttention<0u, true>, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::TileFmhaTraits<true, true, true, true, false, (ck_tile::BlockAttentionBiasEnum)0, false, false, false, false, -1, false> >, ck_tile::BlockFmhaPipelineQXKSVSCustomPolicy<true, true, 3, 3> >, ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, unsigned short, true, true, true, (ck_tile::memory_operation_enum)0>, void> >, ck_tile::FmhaFwdKernel<ck_tile::BlockFmhaPipelineQRKSVSAsync<ck_tile::BlockFmhaPipelineProblem<unsigned short, unsigned short, unsigned short, float, float, unsigned short, unsigned short, float, unsigned short, float, unsigned short, ck_tile::TileFmhaShape<ck_tile::sequence<128, 64, 32, 64, 32, 64>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, true>, true, ck_tile::ComposedAttention<0u, true>, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::TileFmhaTraits<true, true, true, true, false, (ck_tile::BlockAttentionBiasEnum)0, false, false, false, false, -1, false> >, ck_tile::BlockFmhaPipelineQXKSVSCustomPolicy<true, true, 3, 3> >, ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, unsigned short, true, true, true, (ck_tile::memory_operation_enum)0>, void> >::FmhaFwdGroupModeKargs>(ck_tile::FmhaFwdKernel<ck_tile::BlockFmhaPipelineQRKSVSAsync<ck_tile::BlockFmhaPipelineProblem<unsigned short, unsigned short, unsigned short, float, float, unsigned short, unsigned short, float, unsigned short, float, unsigned short, ck_tile::TileFmhaShape<ck_tile::sequence<128, 64, 32, 64, 32, 64>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, true>, true, ck_tile::ComposedAttention<0u, true>, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::TileFmhaTraits<true, true, true, true, false, (ck_tile::BlockAttentionBiasEnum)0, false, false, false, false, -1, false> >, ck_tile::BlockFmhaPipelineQXKSVSCustomPolicy<true, true, 3, 3> >, ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, unsigned short, true, true, true, (ck_tile::memory_operation_enum)0>, void> >, dim3, dim3, unsigned long, ck_tile::FmhaFwdKernel<ck_tile::BlockFmhaPipelineQRKSVSAsync<ck_tile::BlockFmhaPipelineProblem<unsigned short, unsigned short, unsigned short, float, float, unsigned short, unsigned short, float, unsigned short, float, unsigned short, ck_tile::TileFmhaShape<ck_tile::sequence<128, 64, 32, 64, 32, 64>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, true>, true, ck_tile::ComposedAttention<0u, true>, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::TileFmhaTraits<true, true, true, true, false, (ck_tile::BlockAttentionBiasEnum)0, false, false, false, false, -1, false> >, ck_tile::BlockFmhaPipelineQXKSVSCustomPolicy<true, true, 3, 3> >, ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, unsigned short, true, true, true, (ck_tile::memory_operation_enum)0>, void> >::FmhaFwdGroupModeKargs)::{lambda(ck_tile::stream_config const&)#1}&&)\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in float ck_tile::launch_kernel<ck_tile::make_kernel<256, 3, ck_tile::FmhaFwdKernel<ck_tile::BlockFmhaPipelineQRKSVSAsync<ck_tile::BlockFmhaPipelineProblem<unsigned short, unsigned short, unsigned short, float, float, unsigned short, unsigned short, float, unsigned short, float, unsigned short, ck_tile::TileFmhaShape<ck_tile::sequence<128, 64, 32, 64, 32, 64>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, true>, true, ck_tile::ComposedAttention<0u, true>, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::TileFmhaTraits<true, true, true, true, false, (ck_tile::BlockAttentionBiasEnum)0, false, false, false, false, -1, false> >, ck_tile::BlockFmhaPipelineQXKSVSCustomPolicy<true, true, 3, 3> >, ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, unsigned short, true, true, true, (ck_tile::memory_operation_enum)0>, void> >, ck_tile::FmhaFwdKernel<ck_tile::BlockFmhaPipelineQRKSVSAsync<ck_tile::BlockFmhaPipelineProblem<unsigned short, unsigned short, unsigned short, float, float, unsigned short, unsigned short, float, unsigned short, float, unsigned short, ck_tile::TileFmhaShape<ck_tile::sequence<128, 64, 32, 64, 32, 64>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, true>, true, ck_tile::ComposedAttention<0u, true>, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::TileFmhaTraits<true, true, true, true, false, (ck_tile::BlockAttentionBiasEnum)0, false, false, false, false, -1, false> >, ck_tile::BlockFmhaPipelineQXKSVSCustomPolicy<true, true, 3, 3> >, ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, unsigned short, true, true, true, (ck_tile::memory_operation_enum)0>, void> >::FmhaFwdGroupModeKargs>(ck_tile::FmhaFwdKernel<ck_tile::BlockFmhaPipelineQRKSVSAsync<ck_tile::BlockFmhaPipelineProblem<unsigned short, unsigned short, unsigned short, float, float, unsigned short, unsigned short, float, unsigned short, float, unsigned short, ck_tile::TileFmhaShape<ck_tile::sequence<128, 64, 32, 64, 32, 64>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, true>, true, ck_tile::ComposedAttention<0u, true>, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::TileFmhaTraits<true, true, true, true, false, (ck_tile::BlockAttentionBiasEnum)0, false, false, false, false, -1, false> >, ck_tile::BlockFmhaPipelineQXKSVSCustomPolicy<true, true, 3, 3> >, ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, unsigned short, true, true, true, (ck_tile::memory_operation_enum)0>, void> >, dim3, dim3, unsigned long, ck_tile::FmhaFwdKernel<ck_tile::BlockFmhaPipelineQRKSVSAsync<ck_tile::BlockFmhaPipelineProblem<unsigned short, unsigned short, unsigned short, float, float, unsigned short, unsigned short, float, unsigned short, float, unsigned short, ck_tile::TileFmhaShape<ck_tile::sequence<128, 64, 32, 64, 32, 64>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, true>, true, ck_tile::ComposedAttention<0u, true>, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::TileFmhaTraits<true, true, true, true, false, (ck_tile::BlockAttentionBiasEnum)0, false, false, false, false, -1, false> >, ck_tile::BlockFmhaPipelineQXKSVSCustomPolicy<true, true, 3, 3> >, ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, unsigned short, true, true, true, (ck_tile::memory_operation_enum)0>, void> >::FmhaFwdGroupModeKargs)::{lambda(ck_tile::stream_config const&)#1}>(ck_tile::stream_config const&, ck_tile::make_kernel<256, 3, ck_tile::FmhaFwdKernel<ck_tile::BlockFmhaPipelineQRKSVSAsync<ck_tile::BlockFmhaPipelineProblem<unsigned short, unsigned short, unsigned short, float, float, unsigned short, unsigned short, float, unsigned short, float, unsigned short, ck_tile::TileFmhaShape<ck_tile::sequence<128, 64, 32, 64, 32, 64>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, true>, true, ck_tile::ComposedAttention<0u, true>, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::TileFmhaTraits<true, true, true, true, false, (ck_tile::BlockAttentionBiasEnum)0, false, false, false, false, -1, false> >, ck_tile::BlockFmhaPipelineQXKSVSCustomPolicy<true, true, 3, 3> >, ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, unsigned short, true, true, true, (ck_tile::memory_operation_enum)0>, void> >, ck_tile::FmhaFwdKernel<ck_tile::BlockFmhaPipelineQRKSVSAsync<ck_tile::BlockFmhaPipelineProblem<unsigned short, unsigned short, unsigned short, float, float, unsigned short, unsigned short, float, unsigned short, float, unsigned short, ck_tile::TileFmhaShape<ck_tile::sequence<128, 64, 32, 64, 32, 64>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, true>, true, ck_tile::ComposedAttention<0u, true>, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::TileFmhaTraits<true, true, true, true, false, (ck_tile::BlockAttentionBiasEnum)0, false, false, false, false, -1, false> >, ck_tile::BlockFmhaPipelineQXKSVSCustomPolicy<true, true, 3, 3> >, ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, unsigned short, true, true, true, (ck_tile::memory_operation_enum)0>, void> >::FmhaFwdGroupModeKargs>(ck_tile::FmhaFwdKernel<ck_tile::BlockFmhaPipelineQRKSVSAsync<ck_tile::BlockFmhaPipelineProblem<unsigned short, unsigned short, unsigned short, float, float, unsigned short, unsigned short, float, unsigned short, float, unsigned short, ck_tile::TileFmhaShape<ck_tile::sequence<128, 64, 32, 64, 32, 64>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, true>, true, ck_tile::ComposedAttention<0u, true>, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::TileFmhaTraits<true, true, true, true, false, (ck_tile::BlockAttentionBiasEnum)0, false, false, false, false, -1, false> >, ck_tile::BlockFmhaPipelineQXKSVSCustomPolicy<true, true, 3, 3> >, ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, unsigned short, true, true, true, (ck_tile::memory_operation_enum)0>, void> >, dim3, dim3, unsigned long, ck_tile::FmhaFwdKernel<ck_tile::BlockFmhaPipelineQRKSVSAsync<ck_tile::BlockFmhaPipelineProblem<unsigned short, unsigned short, unsigned short, float, float, unsigned short, unsigned short, float, unsigned short, float, unsigned short, ck_tile::TileFmhaShape<ck_tile::sequence<128, 64, 32, 64, 32, 64>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, true>, true, ck_tile::ComposedAttention<0u, true>, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::TileFmhaTraits<true, true, true, true, false, (ck_tile::BlockAttentionBiasEnum)0, false, false, false, false, -1, false> >, ck_tile::BlockFmhaPipelineQXKSVSCustomPolicy<true, true, 3, 3> >, ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, unsigned short, true, true, true, (ck_tile::memory_operation_enum)0>, void> >::FmhaFwdGroupModeKargs)::{lambda(ck_tile::stream_config const&)#1}&&)\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in void grouped_infer_mask_bias_dropout_dispatch<unsigned short, false, false, false, 64, 128>::RunWithKernel<ck_tile::FmhaFwdKernel<ck_tile::BlockFmhaPipelineQRKSVSAsync<ck_tile::BlockFmhaPipelineProblem<unsigned short, unsigned short, unsigned short, float, float, unsigned short, unsigned short, float, unsigned short, float, unsigned short, ck_tile::TileFmhaShape<ck_tile::sequence<128, 64, 32, 64, 32, 64>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, ck_tile::sequence<4, 1, 1>, ck_tile::sequence<32, 32, 16>, true>, true, ck_tile::ComposedAttention<0u, true>, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::TileFmhaTraits<true, true, true, true, false, (ck_tile::BlockAttentionBiasEnum)0, false, false, false, false, -1, false> >, ck_tile::BlockFmhaPipelineQXKSVSCustomPolicy<true, true, 3, 3> >, ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, unsigned short, true, true, true, (ck_tile::memory_operation_enum)0>, void> > >(GroupedForwardParams&, ihipStream_t*)\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in void run_grouped_infer_mask_bias_dropout_dispatch<unsigned short, false, false, false, 64>(GroupedForwardParams&, ihipStream_t*)\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in (anonymous namespace)::efficient_attention_forward_ck(at::Tensor const&, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, std::optional<at::Tensor> const&, std::optional<at::Tensor> const&, std::optional<long>, double, bool, long, std::optional<double>, std::optional<at::Tensor> const&, std::optional<long>, std::optional<at::Tensor> const&, std::optional<long>)\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in c10::impl::make_boxed_from_unboxed_functor<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<std::tuple<at::Tensor, std::optional<at::Tensor>, long, long> (at::Tensor const&, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, std::optional<at::Tensor> const&, std::optional<at::Tensor> const&, std::optional<long>, double, bool, long, std::optional<double>, std::optional<at::Tensor> const&, std::optional<long>, std::optional<at::Tensor> const&, std::optional<long>), &(anonymous namespace)::efficient_attention_forward_ck>, std::tuple<at::Tensor, std::optional<at::Tensor>, long, long>, c10::guts::typelist::typelist<at::Tensor const&, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, std::optional<at::Tensor> const&, std::optional<at::Tensor> const&, std::optional<long>, double, bool, long, std::optional<double>, std::optional<at::Tensor> const&, std::optional<long>, std::optional<at::Tensor> const&, std::optional<long> > >, false>::call(c10::OperatorKernel*, c10::OperatorHandle const&, c10::DispatchKeySet, std::vector<c10::IValue, std::allocator<c10::IValue> >*)\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in c10::Dispatcher::callBoxed(c10::OperatorHandle const&, std::vector<c10::IValue, std::allocator<c10::IValue> >*) const [clone .isra.0]\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in torch::jit::invokeOperatorFromPython(c10::ArrayRef<std::shared_ptr<torch::jit::Operator> >, pybind11::args const&, pybind11::kwargs const&, std::optional<c10::DispatchKey>)\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in torch::jit::_get_operation_for_overload_or_packet(c10::ArrayRef<std::shared_ptr<torch::jit::Operator> >, c10::Symbol, pybind11::args const&, pybind11::kwargs const&, bool, std::optional<c10::DispatchKey>)\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in torch::jit::_get_operation_for_overload_or_packet(std::vector<std::shared_ptr<torch::jit::Operator>, std::allocator<std::shared_ptr<torch::jit::Operator> > > const&, c10::Symbol, pybind11::args const&, pybind11::kwargs const&, bool, std::optional<c10::DispatchKey>)\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in pybind11::cpp_function::initialize<torch::jit::initJITBindings(_object*)::{lambda(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)#222}::operator()(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) const::{lambda(pybind11::args const&, pybind11::kwargs const&)#1}, pybind11::object, pybind11::args const&, pybind11::kwargs const&, pybind11::name, pybind11::doc>(torch::jit::initJITBindings(_object*)::{lambda(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)#222}::operator()(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) const::{lambda(pybind11::args const&, pybind11::kwargs const&)#1}&&, pybind11::object (*)(pybind11::args const&, pybind11::kwargs const&), pybind11::name const&, pybind11::doc const&)::{lambda(pybind11::detail::function_call&)#3}::_FUN(pybind11::detail::function_call&)\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in pybind11::cpp_function::dispatcher(_object*, _object*, _object*)\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyObject_Call\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyEval_EvalFrameDefault\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyObject_FastCallDictTstate\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyObject_Call_Prepend\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyObject_MakeTpCall\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyEval_EvalFrameDefault\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyObject_Call\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyEval_EvalFrameDefault\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyObject_Call\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyEval_EvalFrameDefault\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyObject_FastCallDictTstate\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyObject_Call_Prepend\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyObject_MakeTpCall\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyEval_EvalFrameDefault\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyObject_Call\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyEval_EvalFrameDefault\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyObject_Call\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyEval_EvalFrameDefault\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyObject_FastCallDictTstate\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyObject_Call_Prepend\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyObject_MakeTpCall\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyEval_EvalFrameDefault\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyEval_EvalFrameDefault\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyEval_EvalFrameDefault\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyObject_FastCallDictTstate\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyObject_Call_Prepend\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyObject_MakeTpCall\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyEval_EvalFrameDefault\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyObject_Call\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyEval_EvalFrameDefault\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _PyEval_EvalFrameDefault\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in PyEval_EvalCode\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in PyRun_StringFlags\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in PyRun_SimpleStringFlags\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in Py_RunMain\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in Py_BytesMain\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in _start\n", "pipe": "stderr"}
{"event": "line", "data": "  File \"<unknown>\", line 0, in 0xffffffffffffffff\n", "pipe": "stderr"}