"""Real Hydra integration tests (require hydra-core; no mocked instantiation)."""
from __future__ import annotations
import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
hydra = pytest.importorskip('hydra', reason='Install requirements.txt to run native Hydra integration tests')
import torch
from omegaconf import OmegaConf

from src.components import build_component
from src.hydra_config import compose_config, to_runtime_config
from src.factory import make_model, make_sampler, load_model_from_checkpoint
from src.models.unet import ScoreUNet
from src.models.attention import IdentityAttention
from src.models.examples import TinyScoreNet
from src.sampling.predictors import EulerMaruyama
from src.sampling.correctors import LangevinCorrector
from src.training import train
from src.evaluation import evaluate
from src.cli import evaluation_options
from tests.test_sde import make_sde

ROOT=Path(__file__).resolve().parents[1]


def _run_cli(command, *, cwd=ROOT, timeout=90):
    """Keep quiet on success, but show the actual child-process error on failure."""
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    try:
        result = subprocess.run(
            command, cwd=cwd, check=False, capture_output=True, text=True,
            timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"Command timed out after {timeout}s: {shlex.join(command)}\n"
            f"STDOUT:\n{exc.stdout!r}\nSTDERR:\n{exc.stderr!r}",
            pytrace=False,
        )
    if result.returncode != 0:
        pytest.fail(
            f"Command exited with status {result.returncode}: "
            f"{shlex.join(command)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            pytrace=False,
        )
    return result


def small(overrides=None):
    return compose_config(['experiment=smoke']+list(overrides or []))


def test_hydra_default_graph_matches_direct_model():
    runtime=to_runtime_config(small())
    sde=make_sde(n_fft=30)
    torch.manual_seed(9);wrapped=make_model(runtime,n_fft=30,kernels=sde.kernels.to(torch.complex64))
    torch.manual_seed(9);direct=ScoreUNet(2,8,16,2,dropout=0.,n_fft=30)
    for key,value in direct.state_dict().items():
        torch.testing.assert_close(value,wrapped.score_model.state_dict()[key],rtol=0,atol=0)
    assert runtime['components']['optimizer']['_target_']=='torch.optim.AdamW'


@pytest.mark.parametrize('network',['unet_enhanced','unet_basic','unet_deep','tiny_example'])
def test_network_selection_without_python_dispatch(network):
    # The tiny architecture has different constructor fields, and the smoke
    # experiment intentionally supplies U-Net dimensions; use base config here.
    overrides=[f'model/score_model={network}','logger=csv']
    if network=='tiny_example':
        overrides+=['model.score_model.channels=8','model.score_model.time_dim=16']
    else:
        overrides+=['model.score_model.base_channels=8','model.score_model.time_dim=16',
                    'model.score_model.num_heads=2','model.score_model.dropout=0']
    runtime=to_runtime_config(compose_config(overrides))
    sde=make_sde(n_fft=30)
    model=make_model(runtime,n_fft=30,kernels=sde.kernels.to(torch.complex64))
    x=torch.fft.rfft(torch.randn(2,2,30));y=x.sum(1)
    z=model(x,y,torch.tensor([.1,.8]));z.abs().square().mean().backward()
    assert z.shape==x.shape and torch.isfinite(z).all()
    assert not list(model.score_model.time_embedding.parameters())


def test_nested_components_and_optimizer_scheduler_selection():
    cfg=small(['model/score_model/attention=none','model/score_model/time_embedding=gaussian',
               'model/score_model/features=real_imag','model/optimizer=adam','model/scheduler=cosine',
               'model.score_model.block.kernel_size=5'])
    runtime=to_runtime_config(cfg);sde=make_sde(n_fft=30)
    m=make_model(runtime,n_fft=30,kernels=sde.kernels.to(torch.complex64))
    assert isinstance(m.score_model.attention,IdentityAttention)
    assert m.score_model.enc1a.conv1.kernel_size==(5,)
    assert m.score_model.input_conv.in_channels==6
    optimizer=m.configure_optimizers()
    assert isinstance(optimizer['optimizer'],torch.optim.Adam)
    assert isinstance(optimizer['lr_scheduler']['scheduler'],torch.optim.lr_scheduler.CosineAnnealingLR)


@pytest.mark.parametrize('path', sorted((ROOT/'config/experiment').glob('*.yaml')))
def test_each_experiment_composes(path):
    cfg=compose_config([f'experiment={path.stem}'])
    runtime=to_runtime_config(cfg)
    assert runtime['components']['model']['_target_']=='src.pl_model.DCSModel'
    assert runtime['model']['t_eps']>0


def test_sampler_components_selected_not_only_renamed():
    cfg=small(['model/sampler=pc','model.sampler.num_steps=3','model.sampler.corrector.snr=.05'])
    sampler=make_sampler(to_runtime_config(cfg))
    assert isinstance(sampler.predictor,EulerMaruyama)
    assert isinstance(sampler.corrector,LangevinCorrector)
    assert sampler.corrector.snr==.05 and sampler.num_steps==3


def test_evaluation_does_not_override_trained_model_from_defaults():
    cfg=small(['checkpoint=local.ckpt'])
    # No experiment override is included in the explicitly applied eval keys.
    options=evaluation_options(cfg,['checkpoint'])
    assert options['component_overrides'] is None and not options['overrides']
    cfg=small(['checkpoint=local.ckpt','model/sampler=pc'])
    options=evaluation_options(cfg,['checkpoint','model/sampler'])
    assert options['component_overrides']['sampler']['predictor']['_target_'].endswith('EulerMaruyama')
    with pytest.raises(ValueError):
        evaluation_options(cfg,['model.sde.alpha'])


@pytest.mark.integration
def test_native_training_checkpoint_resume_and_evaluation(tmp_path):
    cfg=small([f'data.data_dir={tmp_path}/data',f'run.output_dir={tmp_path}/runs',
               'run.name=native',f'evaluation.output_dir={tmp_path}/results',
               'model/scheduler=plateau'])
    runtime=to_runtime_config(cfg)
    generator=build_component(runtime['components']['generation'])
    generator(runtime)
    run=train(runtime);checkpoint=run/'checkpoints/last.ckpt'
    with pytest.raises(ValueError,match='target|_target_'):
        load_model_from_checkpoint(checkpoint)
    model,restored,_=load_model_from_checkpoint(checkpoint,trust=True)
    assert restored['components']==runtime['components']
    assert (run/'model_structure.txt').exists() and (run/'components.json').exists()
    pc=to_runtime_config(small(['model/sampler=pc','model.sampler.num_steps=2',
                               'model.sampler.corrector.step_size=.001']))
    _,summary=evaluate(checkpoint,device='cpu',trust=True,
        component_overrides={'sampler':pc['components']['sampler'],'sampling':pc['sampling']})
    assert summary['mean_nfe']==4 and summary['reverse_final_time']==runtime['model']['t_eps']
    resumed=copy.deepcopy(runtime);resumed['trainer']['max_epochs']=2
    resumed['run']['resume']=str(checkpoint);resumed['run']['trust_checkpoint']=True
    train(resumed)
    assert json.loads((run/'artifacts.json').read_text())['global_step']==4


@pytest.mark.integration
def test_native_commandline_smoke(tmp_path):
    common=['experiment=smoke',f'data.data_dir={tmp_path}/data',f'run.output_dir={tmp_path}/runs',
            f'hydra.run.dir={tmp_path}/hydra','run.name=cli_smoke']
    for entry in ('generate_data.py','train.py'):
        _run_cli([sys.executable, str(ROOT / entry), *common])
    checkpoint=tmp_path/'runs/cli_smoke/checkpoints/last.ckpt'
    out=tmp_path/'eval'
    _run_cli([sys.executable,str(ROOT/'evaluate_test_set.py'),f'checkpoint={checkpoint}',
        'evaluation.trust_checkpoint=true','model/sampler=heun','model.sampler.num_steps=2',
        'device=cpu',f'out={out}',f'hydra.run.dir={tmp_path}/eval_hydra'])
    summary=json.loads((out/'summary.json').read_text())
    assert summary['mean_nfe']==4 and summary['num_examples']==4


def test_step_only_override_preserves_checkpoint_pc_recipe():
    from src.factory import edit_sampler
    saved=to_runtime_config(small(['model/sampler=pc','model.sampler.corrector.snr=.08']))
    # The command-line default group is Euler, but only num_steps was requested.
    cli=small(['checkpoint=local.ckpt','model.sampler.num_steps=100'])
    options=evaluation_options(cli,['checkpoint','model.sampler.num_steps'])
    assert 'sampler_edits' in options['component_overrides']
    edit_sampler(saved,options['component_overrides']['sampler_edits'])
    sampler=make_sampler(saved)
    assert isinstance(sampler.predictor,EulerMaruyama)
    assert isinstance(sampler.corrector,LangevinCorrector)
    assert sampler.corrector.snr==.08 and sampler.num_steps==100


@pytest.mark.integration
@pytest.mark.parametrize("entry", [
    "generate_data.py", "train.py", "evaluate_test_set.py", "separate.py",
    "visualize_dataset.py", "visualize_forward_mean.py", "diagnose.py",
])
def test_native_config_discovery_from_another_directory(tmp_path, entry):
    # --cfg exits after composition, without generating data, training or loading
    # checkpoints. A foreign cwd catches accidental use of Path.cwd().
    result = _run_cli(
        [sys.executable, str(ROOT / entry), "experiment=smoke", "--cfg", "job", "--resolve"],
        cwd=tmp_path,
    )
    cfg = OmegaConf.create(result.stdout)
    assert cfg.run.name == "smoke"
    assert cfg.model.sde._target_ == "src.sde.freqsde.ConvMixSDE"


def test_cli_helper_displays_captured_child_error():
    command = [
        sys.executable, "-c",
        "import sys; print('child-stdout-marker', flush=True); "
        "print('child-stderr-marker', file=sys.stderr, flush=True); sys.exit(17)",
    ]
    with pytest.raises(pytest.fail.Exception) as error:
        _run_cli(command)
    message = str(error.value)
    assert "status 17" in message
    assert "child-stdout-marker" in message
    assert "child-stderr-marker" in message
