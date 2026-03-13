# Docker GPU Setup

This document explains how to let the PHASE Docker services see host GPUs and use them for Potts fitting jobs from the webserver.

This document assumes the recommended shared-data workflow from `docs/shared_data_root.md`, where:

- Docker uses `/data/phase` inside containers
- local CLI uses the host path, typically `/scratch/$USER/phase-data`
- both point to the same project tree

## What uses the GPU

Potts PLM fitting runs inside the `worker` container, not in the browser and not in Redis.

In the web UI you can now set the training device explicitly, for example:

- `auto`
- `cpu`
- `cuda`
- `cuda:0`
- `cuda:1`

If Docker is not configured for GPU access, choosing `cuda` in the UI will fail.

## Files in this repository

GPU support is enabled with the optional compose override:

- `docker-compose.yaml`
- `docker-compose.gpu.yaml`

The current override uses CDI GPU devices, not Docker's legacy `--gpus` path.

Use both files together when starting the stack.

## Host prerequisites

You need:

1. NVIDIA driver installed on the host
2. Docker installed
3. NVIDIA Container Toolkit installed and configured for Docker

Typical setup on the host:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify the host runtime first:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
docker run --rm --device nvidia.com/gpu=all ubuntu:24.04 nvidia-smi -L
```

Interpretation:

- if both commands fail, fix Docker/NVIDIA on the host first
- if `--gpus all` fails but the CDI `--device nvidia.com/gpu=all` command works, use the CDI-based PHASE compose override from this repository

## Start PHASE with GPU access

The compose override uses CDI GPU devices through a single variable:

- `PHASE_GPU_CDI_DEVICE=nvidia.com/gpu=all`
- `PHASE_GPU_CDI_DEVICE=nvidia.com/gpu=0`
- `PHASE_GPU_CDI_DEVICE=nvidia.com/gpu=1`

Single GPU example:

```bash
export PHASE_DATA_ROOT=/scratch/$USER/phase-data
./scripts/compose_env.sh
PHASE_GPU_CDI_DEVICE=nvidia.com/gpu=0 \
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up --build -d
```

Multiple GPUs example:

```bash
export PHASE_DATA_ROOT=/scratch/$USER/phase-data
./scripts/compose_env.sh
PHASE_GPU_CDI_DEVICE=nvidia.com/gpu=all \
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up --build -d --scale worker=2
```

Notes:

- `PHASE_GPU_CDI_DEVICE` controls which CDI GPU device is exposed to the containers.
- The current override exposes GPUs to both `backend` and `worker`, but Potts fit jobs run in `worker`.
- If you scale workers, each worker can potentially access the visible GPUs unless you split them manually across services.
- Local `phase_console` should still use:

```bash
export PHASE_DATA_ROOT=/scratch/$USER/phase-data
```

so local CLI and webserver continue to share the same projects.

## Verify GPU visibility inside PHASE containers

Check CUDA visibility in the worker:

```bash
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml exec worker \
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count()); print('torch_cuda=', torch.version.cuda)"
```

Expected result:

- `cuda_available= True`
- `device_count >= 1`

You can also check NVIDIA visibility directly:

```bash
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml exec worker nvidia-smi
```

## Web UI usage

When submitting a Potts fit from the webserver:

- standard fit: set `PLM device` to `cuda:0` or another explicit GPU
- delta fit: set `device` to `cuda:0` or another explicit GPU

If you leave the device as `auto`, PHASE will use CUDA when available and otherwise fall back to CPU.

## Common failure modes

### 1. `nvidia-smi` works on host, but not in container

Cause:

- Docker runtime is not configured for NVIDIA, or the stack was started without `docker-compose.gpu.yaml`

Fix:

1. configure NVIDIA Container Toolkit for Docker
2. restart Docker
3. restart the PHASE stack with:

```bash
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up --build -d
```

### 1b. Legacy `--gpus all` fails with `nvidia-container-cli: mount error: failed to add device rules`

Typical error:

```text
Auto-detected mode as 'legacy'
nvidia-container-cli: mount error: failed to add device rules: ...
```

This is a host-side NVIDIA runtime problem in the legacy GPU path.

On this machine, the practical distinction is:

- `docker run --gpus all ...` may fail
- `docker run --device nvidia.com/gpu=all ...` may still work

If the CDI `--device` command works, use the PHASE CDI override:

```bash
PHASE_GPU_CDI_DEVICE=nvidia.com/gpu=0 \
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up --build -d
```

If **both** commands fail, fix the host runtime first.

Useful host checks:

```bash
nvidia-ctk --version
docker version
docker info
```

If Docker is rootless, configure NVIDIA for rootless Docker. If Docker is rootful, do not set `no-cgroups=true` unless you explicitly know you need it.

### 2. `nvidia-smi` works in container, but `torch.cuda.is_available()` is `False`

Cause:

- the image may not contain a CUDA-capable PyTorch build for the current environment

Check:

```bash
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml exec worker \
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

If CUDA is still unavailable, rebuild the image after ensuring the container installs a CUDA-enabled `torch` build compatible with the host driver.

### 3. Web fit still runs on CPU

Cause:

- device left on `auto` and CUDA is unavailable
- device explicitly set to `cpu`
- worker was started without GPU access

Fix:

1. verify container GPU access with the commands above
2. set device explicitly to `cuda:0`
3. resubmit the job

## Recommended quick check

Use this sequence after any Docker/GPU change:

```bash
docker run --rm --device nvidia.com/gpu=all ubuntu:24.04 nvidia-smi -L
PHASE_GPU_CDI_DEVICE=nvidia.com/gpu=0 docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up --build -d
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml exec worker nvidia-smi -L
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml exec worker \
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda)"
```

If all three steps succeed, webserver Potts fitting can use GPUs.
