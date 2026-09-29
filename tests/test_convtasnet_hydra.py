"""Native config/CLI and real Asteroid tests. Missing dependencies are skipped."""
import copy
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("hydra")
from src.hydra_config import compose_config
from src.baselines.workflow import train, evaluate, restore
from tests.test_convtasnet_baseline import _data

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("overrides", [[], ["baseline_experiment=direct"],
    ["baseline_experiment=legacy_normalization"], ["baseline_experiment=smoke"],
    ["separator/scheduler=plateau"], ["separator/optimizer=adamw"]])
def test_convtasnet_config_composition(overrides):
    cfg = compose_config(overrides, config_name="convtasnet")
    plain = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert plain["separator"]["network"]["_target_"] == "src.baselines.convtasnet.make_convtasnet"
    assert plain["datamodule"]["_target_"] == "src.dataset.MixData_Module"
    if "baseline_experiment=direct" in overrides:
        assert plain["separator"]["postprocess"]["_target_"].endswith("IdentityDeconvolution")
    if "baseline_experiment=legacy_normalization" in overrides:
        assert not plain["separator"]["restore_scale"]


def _run(entry, overrides, cwd):
    cmd = [sys.executable, str(ROOT/entry), *overrides]
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120,
            env={**os.environ, "HYDRA_FULL_ERROR": "1", "OMP_NUM_THREADS": "1"})
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"Timeout: {cmd}\nstdout={exc.stdout}\nstderr={exc.stderr}")
    assert p.returncode == 0, f"{cmd}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"


@pytest.mark.parametrize("entry", ["train_convtasnet.py", "evaluate_convtasnet.py"])
def test_convtasnet_cli_config_from_other_working_directory(tmp_path, entry):
    _run(entry, ["--cfg", "job", "--resolve"], tmp_path)


@pytest.mark.integration
def test_real_asteroid_training_resume_and_evaluation(tmp_path):
    pytest.importorskip("asteroid")
    _data(tmp_path/"data")
    cfg = compose_config(["baseline_experiment=smoke", f"data.data_dir={tmp_path}/data",
        f"run.output_dir={tmp_path}/runs", "run.name=native_baseline", "trainer.accelerator=cpu",
        "logger.tensorboard=null"], config_name="convtasnet")
    cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    folder = train(cfg)
    checkpoint = folder / "checkpoints/last.ckpt"
    assert checkpoint.exists()
    _, result = evaluate(checkpoint, trust=True, device="cpu", batch_size=2,
                         out=tmp_path/"eval")
    assert result["num_examples"] == 4
    model, saved, metadata = restore(checkpoint, trust=True)
    assert model.network.sample_rate == 2000
    assert saved["separator"] == cfg["separator"]
    assert metadata["n_fft"] == 34
    changed = copy.deepcopy(cfg)
    changed["run"].update(resume=str(checkpoint), trust_checkpoint=True)
    changed["trainer"]["max_epochs"] = 2
    train(changed)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["global_step"] == 4


@pytest.mark.integration
def test_real_asteroid_native_train_cli(tmp_path):
    pytest.importorskip("asteroid")
    _data(tmp_path/"data")
    _run("train_convtasnet.py", ["baseline_experiment=smoke", f"data.data_dir={tmp_path}/data",
        f"run.output_dir={tmp_path}/runs", f"hydra.run.dir={tmp_path}/hydra",
        "run.name=baseline_cli", "trainer.accelerator=cpu", "logger.tensorboard=null"], ROOT)
    assert (tmp_path/"runs/baseline_cli/checkpoints/last.ckpt").exists()
