# syntax=docker/dockerfile:1
#
# Strata P100 (sm_60) - self-compiled CUDA 12 engine
# Target: NVIDIA Tesla P100 (compute capability 6.0, Pascal)
# Base:   CUDA 12.8 (CUDA 13 dropped sm_60 offline compilation)
# Target Strata: 0.1.41
#
# The engine is compiled during docker build. On first container start, setup.py
# finds engine-cuda12/BUILD.json (source=local, matching src hash) and reuses it
# instead of recompiling.

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 LANG=C.UTF-8 STRATA_EXECV=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git libatomic1 libgomp1 \
        python3 python3-pip python3-venv unzip wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/strata
COPY . .

ARG CUDA_ARCHITECTURES=60
ARG BUILD_VISION=1

RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt \
    && chmod +x setup.sh docker-entrypoint.sh

RUN .venv/bin/python - <<'PYEOF'
import json, os, pathlib, shutil
import setup

llama = setup.get_llama_cpp()
nvcc, _ = setup.find_nvcc(below=(13, 0))
if nvcc is None:
    raise SystemExit("no CUDA 12.x toolkit found; the sm_60 build needs one")

arch = os.environ.get("CUDA_ARCHITECTURES", "60").strip().strip('"').replace(",", ";")
vision = "gpu" if os.environ.get("BUILD_VISION", "1") == "1" else "none"

# Main engine: STRATA_PORTABLE=ON makes it an AVX2 baseline binary.
# GitHub Actions runners have AVX-512, but the user's CPU (i5-13500H) does not:
# without this, the engine carries AVX-512 code that SIGILLs at runtime.
setup.cmake_build(setup.ROOT, setup.ROOT / "build-cuda12", "strata",
    ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF",
     "-DSTRATA_EXPERIMENTAL_SM60=ON",
     "-DSTRATA_PORTABLE=ON",       # AVX2 baseline: portable to CPUs without AVX-512
     f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
     f"-DCMAKE_CUDA_COMPILER={nvcc}",
     f"-DSTRATA_GGML_DIR={llama}"], None, "build-strata.bat")

if vision != "none":
    setup.cmake_build(setup.ROOT / "tools" / "vision", setup.ROOT / "build-vision-cuda12", "strata-vision",
        [f"-DLLAMA_DIR={llama}", "-DSTRATA_VISION_CUDA=ON",
         "-DSTRATA_PORTABLE=ON",   # same reason: no AVX-512 on the target CPU
         "-DSTRATA_EXPERIMENTAL_SM60=ON",
         f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
         f"-DCMAKE_CUDA_COMPILER={nvcc}"], None, "build-vision.bat")

# 0.1.40: the CUDA 12 engine lives in engine-cuda12/
eng = setup.ROOT / "engine-cuda12"
eng.mkdir(exist_ok=True)
shutil.copy2(setup.ROOT / "build-cuda12" / setup.EXE, eng / setup.EXE)
if vision != "none":
    shutil.copy2(setup.ROOT / "build-vision-cuda12" / "bin" / setup.VEXE, eng / setup.VEXE)

bindir = pathlib.Path(nvcc).parent
meta = {
    "source": "local",
    "version": setup.source_version(),
    "archs": [int(a.split("-")[0]) for a in arch.split(";") if a.split("-")[0].isdigit()],
    "vision": vision,
    "cuda_dirs": [str(d) for d in (bindir, bindir / "x64", bindir.parent / "lib64") if d.is_dir()],
    "src": setup.source_hash(setup.ENGINE_SOURCES),
    "vision_src": setup.source_hash(setup.VISION_SOURCES) if vision != "none" else None,
    "toolkit": 12,
}
(eng / "BUILD.json").write_text(json.dumps(meta, indent=1))
PYEOF

RUN rm -rf build-cuda12 build-vision-cuda12

VOLUME ["/data"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD curl -fs "http://127.0.0.1:${PORT:-8080}/health" || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
