

export WANDB_DISABLED=true
export WANDB_MODE=disabled


export MILABENCH_GPU_ARCH=rocm
export MILABENCH_WORDIR="/data/output/$MILABENCH_GPU_ARCH"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"
export MILABENCH_SIZER_SAVE="$MILABENCH_WORDIR/scaling.yaml"
export MILABENCH_CONFIG=/home/amd/milabench/config/all.yaml
export MILABENCH_VENV=/home/amd/milabench/.venv/bin/activate

export PYTHONPATH=/home/amd/milabench/.venv/lib/python3.12/site-packages/_rocm_sdk_core/share/amd_smi
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE


mkdir -p $MILABENCH_BASE

(
        . $MILABENCH_VENV
        milabench pin --variant rocm --from-scratch
        milabench install
        milabench prepare
        milabench run
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



