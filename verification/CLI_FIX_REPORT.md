# Hydra command-line config-discovery fix

## Scope

Only `src/cli.py` and `tests/test_hydra_workflow.py` are modified in the
application/test code. No model, SDE, loss, optimizer, sampler, dataset or YAML
configuration is changed. Existing datasets/checkpoints do not need conversion.

## Identified defect

All seven Hydra-decorated functions are defined in the imported `src.cli`
module and used `config_path='../config'`. Hydra 1.3.2's module-based resolution
turns this into `pkg://config`. The project's root `config` directory is not an
importable config package. By contrast, `compose_config()` already uses
`initialize_config_dir()` with an absolute filesystem path, explaining why
programmatic composition can pass while command-line launch fails.

Each decorator now uses `CONFIG_DIR`, defined as:

```python
CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "config")
```

This keeps native Hydra groups/overrides/multirun; it does not replace Hydra.
It also avoids reliance on the process working directory.

The user's reported subprocess failure did not include captured stderr, so its
exact child error has not been confirmed. This is a concrete defect identified
in the supplied code, not a claim to have reproduced the user's full execution.

## Test diagnostics and regression coverage

The existing CLI smoke test now uses a helper that still fails on nonzero exit
and on timeout, but prints the command, captured stdout and captured stderr.
HYDRA_FULL_ERROR=1 is supplied to subprocesses.

Seven added native tests compose configuration through the real root entry
points from another working directory; a separate test checks the subprocess
helper displays captured errors. No failure is skipped or tolerated.

## Local verification

- `python -m compileall -q src tests *.py`: succeeded.
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q --disable-warnings`:
  **48 passed, 1 skipped, 11 warnings** (8.53 seconds).
- The skipped module is `tests/test_hydra_workflow.py`, because hydra-core is not
  installed in this container. Installation attempts could not access the
  package repository. Native CLI execution is therefore NOT claimed tested.
- The user's own run reported **92 passed, 1 failed** before this patch; that is
  distinct from the local result above.
- No CUDA or long-training test was run.

Run in the existing environment that already has Hydra:

```bash
python -m pytest tests/test_hydra_workflow.py::test_native_commandline_smoke -q
python -m pytest tests/test_hydra_workflow.py -q
python -m pytest -q
```

Inspect configuration without generating or training:

```bash
python generate_data.py experiment=smoke --cfg job --resolve
```

## Sources for the config-path and subprocess behavior

- https://github.com/facebookresearch/hydra/blob/v1.3.2/hydra/_internal/utils.py
- https://hydra.cc/docs/1.3/advanced/search_path/
- https://docs.python.org/3/library/subprocess.html
