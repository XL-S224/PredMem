# Execution environments

The frozen requirement files are exported from the warm environments that ran the experiments. They intentionally preserve the working versions even where upstream package metadata would now cause the resolver to choose something else. Never install both backends' freezes into one environment.

For ESM3, start with Python3.12 and CUDA12.1-compatible driver:

```bash
python3.12 -m venv .venv-esm3
source .venv-esm3/bin/activate
python -m pip install 'setuptools==80.10.2' wheel
python -m pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchtext==0.18.0 --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-deps -r environments/esm3-frozen.txt
python -m pip install --no-deps -e upstream/esm
python -m pip install --no-deps -e .
```

For DPLM and evaluation, start with Python3.10:

```bash
python3.10 -m venv .venv-dplm
source .venv-dplm/bin/activate
python -m pip install 'setuptools<81' wheel ninja
python -m pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip install 'torch_scatter==2.1.2+pt22cu121' --find-links https://data.pyg.org/whl/torch-2.2.0+cu121.html
python -m pip install --no-deps -r environments/dplm2-frozen.txt
python -m pip install --no-deps -e upstream/dplm
python -m pip install --no-deps --no-build-isolation -e upstream/dplm/vendor/openfold
python -m pip install --no-deps -e .
```

OpenFold's extension build requires a matching CUDA toolkit/compiler. The freezes include dependency packages explicitly; `--no-deps` preserves this tested snapshot instead of re-resolving it. Both environments require `PyYAML==6.0.3`.

Native ESM3 and fair-esm share an import name. The YAML launcher adds the ESM3 source only for the ESM3 backend. The evaluation subprocess uses the DPLM environment and its own OpenFold path.

Weights are authorized upstream assets, not vendored code. Use `PM_HF_HOME` and `PM_TORCH_HOME` to choose caches; export `PM_EVAL_PYTHON` to the evaluation environment interpreter when generating from ESM3. Offline runs must set both YAML runtime environment flags to1 and have all required weights already cached.

## Foldseek

For the paper benchmark, use the build for your platform from the [Foldseek 10.941cd33 release](https://github.com/steineggerlab/foldseek/releases/tag/10-941cd33). Use `foldseek-linux-avx2.tar.gz` on Linux x86-64 with AVX2. Add the extracted `bin/` directory to `PATH` or set `PM_FOLDSEEK` to the full executable path.

PredMem records the executable version and SHA-256 in the clustering output. To enforce a particular version or build, set `evaluation.foldseek_version` or `evaluation.foldseek_sha256` in `configs/base.yaml`.
