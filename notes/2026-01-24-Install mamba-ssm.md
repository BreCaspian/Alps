# mamba-ssm Install

## My Computer Info

```bash
(torch) 12:47 😀 yao@robot:~$ nproc
20

(torch) 12:53 😀 yao@robot:~$ free -h
               total        used        free      shared  buff/cache   available
Mem:            15Gi       3.4Gi       9.3Gi        38Mi       2.7Gi        11Gi
Swap:          7.4Gi       2.8Gi       4.6Gi
````

---

## Install Command

### Option 1: Fixed 4 Jobs (Safe)

```bash
export MAX_JOBS=4
export NINJA_FLAGS="-j4"
python -m pip install -U mamba-ssm --no-build-isolation --no-cache-dir -v
```

### Option 2: Use All Cores (Not Recommended)

> ⚠️ Might freeze the system / VSCode due to high CPU usage.

```bash
export MAX_JOBS=$(nproc)
export NINJA_FLAGS="-j$(nproc)"
python -m pip install -U mamba-ssm --no-build-isolation --no-cache-dir -v
```

### Option 3: Use Half Cores (Recommended)

```bash
JOBS=$(( $(nproc) / 2 ))
[ "$JOBS" -lt 1 ] && JOBS=1
export MAX_JOBS=$JOBS
export NINJA_FLAGS="-j$JOBS"
python -m pip install -U mamba-ssm --no-build-isolation --no-cache-dir -v
```

---


















### log 

````md

  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi8ELi1ELb1ELb0ELb0ELb1EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 80 registers, used 1 barriers
  ptxas info    : Compile time = 14.748 ms
  ptxas info    : Compiling entry function '_Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi8ELi1ELb1ELb0ELb1ELb0EffEEv13SSMParamsBase' for 'sm_121'
  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi8ELi1ELb1ELb0ELb1ELb0EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 86 registers, used 1 barriers
  ptxas info    : Compile time = 13.946 ms
  ptxas info    : Compiling entry function '_Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi8ELi1ELb1ELb0ELb1ELb1EffEEv13SSMParamsBase' for 'sm_121'
  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi8ELi1ELb1ELb0ELb1ELb1EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 88 registers, used 1 barriers
  ptxas info    : Compile time = 16.020 ms
  ptxas info    : Compiling entry function '_Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi8ELi1ELb1ELb1ELb0ELb0EffEEv13SSMParamsBase' for 'sm_121'
  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi8ELi1ELb1ELb1ELb0ELb0EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 85 registers, used 1 barriers
  ptxas info    : Compile time = 14.166 ms
  ptxas info    : Compiling entry function '_Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi8ELi1ELb1ELb1ELb0ELb1EffEEv13SSMParamsBase' for 'sm_121'
  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi8ELi1ELb1ELb1ELb0ELb1EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 87 registers, used 1 barriers
  ptxas info    : Compile time = 17.107 ms
  ptxas info    : Compiling entry function '_Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi8ELi1ELb1ELb1ELb1ELb0EffEEv13SSMParamsBase' for 'sm_121'
  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi8ELi1ELb1ELb1ELb1ELb0EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 91 registers, used 1 barriers
  ptxas info    : Compile time = 14.346 ms
  ptxas info    : Compiling entry function '_Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi8ELi1ELb1ELb1ELb1ELb1EffEEv13SSMParamsBase' for 'sm_121'
  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi8ELi1ELb1ELb1ELb1ELb1EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 93 registers, used 1 barriers
  ptxas info    : Compile time = 17.350 ms
  ptxas info    : Compiling entry function '_Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb0ELb0ELb0ELb0EffEEv13SSMParamsBase' for 'sm_121'
  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb0ELb0ELb0ELb0EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 70 registers, used 1 barriers
  ptxas info    : Compile time = 12.493 ms
  ptxas info    : Compiling entry function '_Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb0ELb0ELb0ELb1EffEEv13SSMParamsBase' for 'sm_121'
  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb0ELb0ELb0ELb1EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 72 registers, used 1 barriers
  ptxas info    : Compile time = 14.136 ms
  ptxas info    : Compiling entry function '_Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb0ELb0ELb1ELb0EffEEv13SSMParamsBase' for 'sm_121'
  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb0ELb0ELb1ELb0EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 70 registers, used 1 barriers
  ptxas info    : Compile time = 12.682 ms
  ptxas info    : Compiling entry function '_Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb0ELb0ELb1ELb1EffEEv13SSMParamsBase' for 'sm_121'
  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb0ELb0ELb1ELb1EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 71 registers, used 1 barriers
  ptxas info    : Compile time = 14.895 ms
  ptxas info    : Compiling entry function '_Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb0ELb1ELb0ELb0EffEEv13SSMParamsBase' for 'sm_121'
  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb0ELb1ELb0ELb0EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 72 registers, used 1 barriers
  ptxas info    : Compile time = 13.282 ms
  ptxas info    : Compiling entry function '_Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb0ELb1ELb0ELb1EffEEv13SSMParamsBase' for 'sm_121'
  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb0ELb1ELb0ELb1EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 72 registers, used 1 barriers
  ptxas info    : Compile time = 14.877 ms
  ptxas info    : Compiling entry function '_Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb0ELb1ELb1ELb0EffEEv13SSMParamsBase' for 'sm_121'
  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb0ELb1ELb1ELb0EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 72 registers, used 1 barriers
  ptxas info    : Compile time = 13.358 ms
  ptxas info    : Compiling entry function '_Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb0ELb1ELb1ELb1EffEEv13SSMParamsBase' for 'sm_121'
  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb0ELb1ELb1ELb1EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 79 registers, used 1 barriers
  ptxas info    : Compile time = 15.391 ms
  ptxas info    : Compiling entry function '_Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb1ELb0ELb0ELb0EffEEv13SSMParamsBase' for 'sm_121'
  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb1ELb0ELb0ELb0EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 61 registers, used 1 barriers
  ptxas info    : Compile time = 8.880 ms
  ptxas info    : Compiling entry function '_Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb1ELb0ELb0ELb1EffEEv13SSMParamsBase' for 'sm_121'
  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb1ELb0ELb0ELb1EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 69 registers, used 1 barriers
  ptxas info    : Compile time = 9.878 ms
  ptxas info    : Compiling entry function '_Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb1ELb0ELb1ELb0EffEEv13SSMParamsBase' for 'sm_121'
  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb1ELb0ELb1ELb0EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 64 registers, used 1 barriers
  ptxas info    : Compile time = 9.831 ms
  ptxas info    : Compiling entry function '_Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb1ELb0ELb1ELb1EffEEv13SSMParamsBase' for 'sm_121'
  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb1ELb0ELb1ELb1EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 72 registers, used 1 barriers
  ptxas info    : Compile time = 10.776 ms
  ptxas info    : Compiling entry function '_Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb1ELb1ELb0ELb0EffEEv13SSMParamsBase' for 'sm_121'
  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb1ELb1ELb0ELb0EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 64 registers, used 1 barriers
  ptxas info    : Compile time = 10.267 ms
  ptxas info    : Compiling entry function '_Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb1ELb1ELb0ELb1EffEEv13SSMParamsBase' for 'sm_121'
  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb1ELb1ELb0ELb1EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 72 registers, used 1 barriers
  ptxas info    : Compile time = 10.952 ms
  ptxas info    : Compiling entry function '_Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb1ELb1ELb1ELb0EffEEv13SSMParamsBase' for 'sm_121'
  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb1ELb1ELb1ELb0EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 72 registers, used 1 barriers
  ptxas info    : Compile time = 9.958 ms
  ptxas info    : Compiling entry function '_Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb1ELb1ELb1ELb1EffEEv13SSMParamsBase' for 'sm_121'
  ptxas info    : Function properties for _Z25selective_scan_fwd_kernelI32Selective_Scan_fwd_kernel_traitsILi32ELi4ELi1ELb1ELb1ELb1ELb1EffEEv13SSMParamsBase
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 80 registers, used 1 barriers
  ptxas info    : Compile time = 10.905 ms
  g++ -pthread -B /home/yao/miniconda3/envs/torch/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/yao/miniconda3/envs/torch/include -fPIC -O2 -isystem /home/yao/miniconda3/envs/torch/include -shared /tmp/pip-install-7ls2njmk/mamba-ssm_1e82044043e3438094c74f84fd567513/build/temp.linux-x86_64-cpython-310/csrc/selective_scan/selective_scan.o /tmp/pip-install-7ls2njmk/mamba-ssm_1e82044043e3438094c74f84fd567513/build/temp.linux-x86_64-cpython-310/csrc/selective_scan/selective_scan_bwd_bf16_complex.o /tmp/pip-install-7ls2njmk/mamba-ssm_1e82044043e3438094c74f84fd567513/build/temp.linux-x86_64-cpython-310/csrc/selective_scan/selective_scan_bwd_bf16_real.o /tmp/pip-install-7ls2njmk/mamba-ssm_1e82044043e3438094c74f84fd567513/build/temp.linux-x86_64-cpython-310/csrc/selective_scan/selective_scan_bwd_fp16_complex.o /tmp/pip-install-7ls2njmk/mamba-ssm_1e82044043e3438094c74f84fd567513/build/temp.linux-x86_64-cpython-310/csrc/selective_scan/selective_scan_bwd_fp16_real.o /tmp/pip-install-7ls2njmk/mamba-ssm_1e82044043e3438094c74f84fd567513/build/temp.linux-x86_64-cpython-310/csrc/selective_scan/selective_scan_bwd_fp32_complex.o /tmp/pip-install-7ls2njmk/mamba-ssm_1e82044043e3438094c74f84fd567513/build/temp.linux-x86_64-cpython-310/csrc/selective_scan/selective_scan_bwd_fp32_real.o /tmp/pip-install-7ls2njmk/mamba-ssm_1e82044043e3438094c74f84fd567513/build/temp.linux-x86_64-cpython-310/csrc/selective_scan/selective_scan_fwd_bf16.o /tmp/pip-install-7ls2njmk/mamba-ssm_1e82044043e3438094c74f84fd567513/build/temp.linux-x86_64-cpython-310/csrc/selective_scan/selective_scan_fwd_fp16.o /tmp/pip-install-7ls2njmk/mamba-ssm_1e82044043e3438094c74f84fd567513/build/temp.linux-x86_64-cpython-310/csrc/selective_scan/selective_scan_fwd_fp32.o -L/home/yao/miniconda3/envs/torch/lib/python3.10/site-packages/torch/lib -L/usr/local/cuda/lib64 -lc10 -ltorch -ltorch_cpu -ltorch_python -lcudart -lc10_cuda -ltorch_cuda -o build/lib.linux-x86_64-cpython-310/selective_scan_cuda.cpython-310-x86_64-linux-gnu.so
  installing to build/bdist.linux-x86_64/wheel
  running install
  running install_lib
  creating build/bdist.linux-x86_64/wheel
  creating build/bdist.linux-x86_64/wheel/mamba_ssm
  creating build/bdist.linux-x86_64/wheel/mamba_ssm/utils
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/utils/torch.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/utils
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/utils/generation.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/utils
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/utils/__init__.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/utils
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/utils/determinism.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/utils
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/utils/hf.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/utils
  creating build/bdist.linux-x86_64/wheel/mamba_ssm/models
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/models/mixer_seq_simple.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/models
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/models/__init__.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/models
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/models/config_mamba.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/models
  creating build/bdist.linux-x86_64/wheel/mamba_ssm/distributed
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/distributed/distributed_utils.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/distributed
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/distributed/tensor_parallel.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/distributed
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/distributed/__init__.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/distributed
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/__init__.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm
  creating build/bdist.linux-x86_64/wheel/mamba_ssm/modules
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/modules/block.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/modules
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/modules/mlp.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/modules
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/modules/mamba2_simple.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/modules
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/modules/mha.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/modules
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/modules/__init__.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/modules
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/modules/ssd_minimal.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/modules
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/modules/mamba2.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/modules
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/modules/mamba_simple.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/modules
  creating build/bdist.linux-x86_64/wheel/mamba_ssm/ops
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/ops/selective_scan_interface.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/ops
  creating build/bdist.linux-x86_64/wheel/mamba_ssm/ops/triton
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/ops/triton/layernorm_gated.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/ops/triton
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/ops/triton/ssd_bmm.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/ops/triton
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/ops/triton/ssd_chunk_scan.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/ops/triton
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/ops/triton/layer_norm.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/ops/triton
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/ops/triton/__init__.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/ops/triton
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/ops/triton/softplus.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/ops/triton
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/ops/triton/ssd_combined.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/ops/triton
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/ops/triton/ssd_chunk_state.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/ops/triton
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/ops/triton/k_activations.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/ops/triton
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/ops/triton/ssd_state_passing.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/ops/triton
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/ops/triton/selective_state_update.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/ops/triton
  copying build/lib.linux-x86_64-cpython-310/mamba_ssm/ops/__init__.py -> build/bdist.linux-x86_64/wheel/./mamba_ssm/ops
  copying build/lib.linux-x86_64-cpython-310/selective_scan_cuda.cpython-310-x86_64-linux-gnu.so -> build/bdist.linux-x86_64/wheel/.
  running install_egg_info
  Copying mamba_ssm.egg-info to build/bdist.linux-x86_64/wheel/./mamba_ssm-2.3.0-py3.10.egg-info
  running install_scripts
  creating build/bdist.linux-x86_64/wheel/mamba_ssm-2.3.0.dist-info/WHEEL
  creating '/tmp/pip-ephem-wheel-cache-hi226qbw/wheels/f4/a4/5d/c4c3dbde9d7a976e15c6c1a3296110ffd4befb4dd5e10fa20f/tmpp5p4usoe/.tmp-el_okok2/mamba_ssm-2.3.0-cp310-cp310-linux_x86_64.whl' and adding 'build/bdist.linux-x86_64/wheel' to it
  adding 'selective_scan_cuda.cpython-310-x86_64-linux-gnu.so'
  adding 'mamba_ssm/__init__.py'
  adding 'mamba_ssm/distributed/__init__.py'
  adding 'mamba_ssm/distributed/distributed_utils.py'
  adding 'mamba_ssm/distributed/tensor_parallel.py'
  adding 'mamba_ssm/models/__init__.py'
  adding 'mamba_ssm/models/config_mamba.py'
  adding 'mamba_ssm/models/mixer_seq_simple.py'
  adding 'mamba_ssm/modules/__init__.py'
  adding 'mamba_ssm/modules/block.py'
  adding 'mamba_ssm/modules/mamba2.py'
  adding 'mamba_ssm/modules/mamba2_simple.py'
  adding 'mamba_ssm/modules/mamba_simple.py'
  adding 'mamba_ssm/modules/mha.py'
  adding 'mamba_ssm/modules/mlp.py'
  adding 'mamba_ssm/modules/ssd_minimal.py'
  adding 'mamba_ssm/ops/__init__.py'
  adding 'mamba_ssm/ops/selective_scan_interface.py'
  adding 'mamba_ssm/ops/triton/__init__.py'
  adding 'mamba_ssm/ops/triton/k_activations.py'
  adding 'mamba_ssm/ops/triton/layer_norm.py'
  adding 'mamba_ssm/ops/triton/layernorm_gated.py'
  adding 'mamba_ssm/ops/triton/selective_state_update.py'
  adding 'mamba_ssm/ops/triton/softplus.py'
  adding 'mamba_ssm/ops/triton/ssd_bmm.py'
  adding 'mamba_ssm/ops/triton/ssd_chunk_scan.py'
  adding 'mamba_ssm/ops/triton/ssd_chunk_state.py'
  adding 'mamba_ssm/ops/triton/ssd_combined.py'
  adding 'mamba_ssm/ops/triton/ssd_state_passing.py'
  adding 'mamba_ssm/utils/__init__.py'
  adding 'mamba_ssm/utils/determinism.py'
  adding 'mamba_ssm/utils/generation.py'
  adding 'mamba_ssm/utils/hf.py'
  adding 'mamba_ssm/utils/torch.py'
  adding 'mamba_ssm-2.3.0.dist-info/licenses/AUTHORS'
  adding 'mamba_ssm-2.3.0.dist-info/licenses/LICENSE'
  adding 'mamba_ssm-2.3.0.dist-info/METADATA'
  adding 'mamba_ssm-2.3.0.dist-info/WHEEL'
  adding 'mamba_ssm-2.3.0.dist-info/top_level.txt'
  adding 'mamba_ssm-2.3.0.dist-info/RECORD'
  removing build/bdist.linux-x86_64/wheel
  Building wheel for mamba-ssm (pyproject.toml) ... done
  Created wheel for mamba-ssm: filename=mamba_ssm-2.3.0-cp310-cp310-linux_x86_64.whl size=349660528 sha256=f9d55bfcbb7576a7ff66389c0659952f689a832200f34c4bc6bbc578a7d769b4
  Stored in directory: /tmp/pip-ephem-wheel-cache-hi226qbw/wheels/f4/a4/5d/c4c3dbde9d7a976e15c6c1a3296110ffd4befb4dd5e10fa20f
Successfully built mamba-ssm
Installing collected packages: regex, huggingface-hub, tokenizers, transformers, mamba-ssm
  Attempting uninstall: huggingface-hub
    Found existing installation: huggingface_hub 1.3.3
    Uninstalling huggingface_hub-1.3.3:
      Removing file or directory /home/yao/miniconda3/envs/torch/bin/hf
      Removing file or directory /home/yao/miniconda3/envs/torch/bin/tiny-agents
      Removing file or directory /home/yao/miniconda3/envs/torch/lib/python3.10/site-packages/huggingface_hub-1.3.3.dist-info/
      Removing file or directory /home/yao/miniconda3/envs/torch/lib/python3.10/site-packages/huggingface_hub/
      Successfully uninstalled huggingface_hub-1.3.3
  changing mode of /home/yao/miniconda3/envs/torch/bin/hf to 775
  changing mode of /home/yao/miniconda3/envs/torch/bin/huggingface-cli to 775
  changing mode of /home/yao/miniconda3/envs/torch/bin/tiny-agents to 775
  changing mode of /home/yao/miniconda3/envs/torch/bin/transformers to 775
  changing mode of /home/yao/miniconda3/envs/torch/bin/transformers-cli to 775
Successfully installed huggingface-hub-0.36.0 mamba-ssm-2.3.0 regex-2026.1.15 tokenizers-0.22.2 transformers-4.57.6
(torch) 12:45 😀 yao@robot:~$ python -c "import mamba_ssm; print('mamba_ssm import OK')"
mamba_ssm import OK
(torch) 12:47 😀 yao@robot:~$ 

