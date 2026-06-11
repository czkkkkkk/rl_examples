# GRPO Training on Neuron — Setup Guideline

This guide walks through setting up an environment for GRPO training on AWS Neuron (Trainium) instances.

## Step 1: Create a Python environment

Create a conda environment with **Python 3.12**:

```bash
conda create -n test_eager python=3.12 -y
conda activate test_eager
```

## Step 2: Install Neuron runtime packages

### 2.1 Configure the Neuron apt repository

```bash
. /etc/os-release
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

# Update OS packages
sudo apt-get update -y
```

> Note: `apt-key` is deprecated on newer Ubuntu releases and will print a warning, but it still works.

### 2.2 Install the Neuron driver, collectives, and runtime library

Install the following pinned versions:

| Package | Version |
|---|---|
| `aws-neuronx-dkms` | `2.28.0.0` |
| `aws-neuronx-collectives` | `2.32.28.0-452cba8de` |
| `aws-neuronx-runtime-lib` | `2.32.31.0-0234f5ed2` |

```bash
sudo apt-get install -y \
    aws-neuronx-dkms=2.28.0.0 \
    aws-neuronx-collectives=2.32.28.0-452cba8de \
    aws-neuronx-runtime-lib=2.32.31.0-0234f5ed2
```

Verify the installed versions:

```bash
dpkg -l | grep -i neuron
```

Expected output:

```
ii  aws-neuronx-collectives    2.32.28.0-452cba8de    amd64    neuron_ccom built using CMake
ii  aws-neuronx-dkms           2.28.0.0               all      aws-neuronx driver in DKMS format.
ii  aws-neuronx-runtime-lib    2.32.31.0-0234f5ed2    amd64    neuron_runtime built using CMake
```

## Step 3: Install the Neuron compiler (neuronx-cc)

Activate the conda environment, then point pip at the Neuron package repository and install the pinned compiler version:

```bash
conda activate test_eager

python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
python -m pip install neuronx-cc==2.25.3371.0+f524f7f8
```

Verify the installation:

```bash
neuronx-cc --version
```

Expected output:

```
NeuronX Compiler version 2.25.3371.0+f524f7f8

Python version 3.12.13
HWM version 2.25.0.3371++f524f7f8
NumPy version 2.4.6
```

## Step 4: Install torch_neuronx from a local wheel

The `torch_neuronx` wheel is provided manually (not from the pip repository):

```
torch_neuronx-2.11.3.0.19138+4dee388.dev-cp312-cp312-linux_x86_64.whl
```

Install it into the conda environment (this also pulls in `torch 2.11.0` as a dependency):

```bash
conda activate test_eager

python -m pip install ./torch_neuronx-2.11.3.0.19138+4dee388.dev-cp312-cp312-linux_x86_64.whl
```

Verify the installation:

```bash
python -c "import torch_neuronx; import torch; print('torch_neuronx OK, torch', torch.__version__)"
pip list | grep -i torch
```

Expected output:

```
torch_neuronx OK, torch 2.11.0+cu130
torch                  2.11.0
torch-neuronx          2.11.3.0.19138+4dee388.dev
```

Run a simple computation on the Neuron device to confirm the device works end to end:

```bash
python -c "
import torch
import torch_neuronx
a = torch.ones([3], device='neuron')
print(a + 1)
"
```

Expected output:

```
tensor([2., 2., 2.], device='neuron:0')
```

> Note: warnings about overridden operator kernels and synchronous `nrta_tensor_write`/`nrta_tensor_read` are expected and harmless.

## Step 5: Install transformers, trl, and accelerate (editable)

Clone the `neuron` branches of the forked `transformers` and `trl` repositories, plus the latest `main` branch of the official `accelerate` repository, and install all three in editable mode. Install `accelerate` last so it replaces the released version pulled in as a `trl` dependency:

```bash
conda activate test_eager

git clone -b neuron https://github.com/czkkkkkk/transformers.git
git clone -b neuron https://github.com/czkkkkkk/trl.git
git clone https://github.com/huggingface/accelerate.git

python -m pip install -e ./transformers
python -m pip install -e ./trl
python -m pip install -e ./accelerate
```

Verify the installation:

```bash
python -c "
import transformers, trl, accelerate
print('transformers', transformers.__version__)
print('trl', trl.__version__)
print('accelerate', accelerate.__version__)
"
```

Expected output:

```
transformers 5.3.0.dev0
trl 1.0.0.dev0
accelerate 1.14.0.dev0
```

## Step 6: Install remaining Python dependencies

The GRPO example needs a few extra packages (server stack, reward verification, logging) that are not pulled in by the editable installs:

```bash
python -m pip install \
    fastapi==0.136.3 uvicorn==0.49.0 pydantic==2.13.4 \
    pyzmq==27.1.0 msgpack==1.1.2 \
    math-verify==0.9.0 latex2sympy2_extended==1.11.0 \
    tensorboard==2.20.0 pillow==12.2.0
```

## Step 7: Run the GRPO example (Qwen3-8B, FSDP=16 training + DP=16 rollout)

Clone the examples repository:

```bash
git clone https://github.com/czkkkkkk/rl_examples
cd rl_examples/deepmath
```

Before running, adjust the scripts for your environment:

1. In both `hf_serve/run_hf_serve_qwen8b_dp16tp1.sh` and `run_neuron.sh`, point the `source .../bin/activate` line at your conda env (e.g. `test_eager`).
2. Make sure the NEFF cache dirs (`TORCH_NEURONX_NEFF_CACHE_DIR`, `TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR`) point to writable paths.

The run uses 32 Neuron cores on a trn2.48xlarge: training (FSDP=16) pins cores 0-15, the rollout server (DP=16) pins cores 16-31.

**Terminal 1 — start the rollout server first:**

```bash
bash hf_serve/run_hf_serve_qwen8b_dp16tp1.sh
```

Wait until all workers are ready and uvicorn is listening:

```
[HF-Serve] worker 7 ready (16/16)
INFO:     Uvicorn running on http://127.0.0.1:30000
```

(You can check readiness with `curl http://127.0.0.1:30000/health` → 200.)

**Terminal 2 — start GRPO training:**

```bash
bash run_neuron.sh
```

The first step is slow (~340 s) due to NEFF compilation; steady-state steps take ~180 s. A successful 10-step run ends with:

```
{'train_runtime': '2115', 'train_samples_per_second': '0.076', 'train_steps_per_second': '0.005', ...}
100%|██████████| 10/10 [35:15<00:00, 211.51s/it]
```

### Fixes applied during validation

These changes were required to make the run work:

- **rl_examples** (applied to the local checkout): removed `--tp-size 1` from `hf_serve/run_hf_serve_qwen8b_dp16tp1.sh`, removed `use_nkipy: false` from `grpo_configs/grpo_hf_serve_qwen8b.yaml`, and removed the `config.use_nkipy` check in `main.py` — the published trl `neuron` branch has no nkipy/TP support.
- **trl (neuron branch)**: `trl/models/utils.py` was missing the per-layer FSDP2 wrapping and rollout unshard logic from the original `dev_grpo` branch. Without it, FSDP puts all params in one group and Neuron fails with `NRT EXECUTION FAILED ... Failed to allocate resource` while moving states to device. Re-applied the `dev_grpo` version of `trl/models/utils.py` (needs to be pushed to the `neuron` branch).
- **accelerate gather fix (runtime patch in `main.py`)**: accelerate main's `_gpu_gather_one` passes the original tensor into `all_gather_into_tensor`; torch_neuronx's all_gather lowering indexes `tensor.shape[0]`, so 0-d reward tensors crash with `IndexError: tuple index out of range`. `main.py` monkey-patches `accelerate.utils.operations._gpu_gather` at startup (gated on `ON_NEURON=1`) to gather flattened `t.view(-1)` tensors — no accelerate source change needed.
