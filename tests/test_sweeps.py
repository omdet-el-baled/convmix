from pathlib import Path
import yaml
import pytest

from src.config import load_config,validate_config
from src.sweeps import run_sweep,training_identity
from src.data.generate import generate_dataset


def test_all_experiment_configs_validate():
    root=Path(__file__).resolve().parents[1]
    for p in sorted((root/'configs/experiments').glob('*.yaml')):
        validate_config(load_config(p))


def test_budget_changes_training_identity(tiny_config):
    import copy
    longer=copy.deepcopy(tiny_config)
    longer['trainer']['max_epochs']+=1
    assert training_identity(tiny_config)!=training_identity(longer)


@pytest.mark.integration
def test_short_sweep_reuses_only_matching_finished_runs(tiny_config,tmp_path):
    cfg=tiny_config;generate_dataset(cfg)
    config_file=tmp_path/'tiny.yaml';config_file.write_text(yaml.safe_dump(cfg))
    spec={'mode':'train','base_config':str(config_file),'output_dir':str(tmp_path/'sweep'),
          'seeds':[42],'grid':{'model.mismatch_probability':[0.0]},
          'evaluation_cases':[{'sampling.solver':'euler','sampling.num_steps':2}]}
    sweep_file=tmp_path/'sweep.yaml';sweep_file.write_text(yaml.safe_dump(spec))
    run_sweep(sweep_file,execute=False)
    assert not (tmp_path/'sweep').exists()
    run_sweep(sweep_file,execute=True)
    result=tmp_path/'sweep/results_per_run.csv'
    assert result.exists()
    before=result.read_text()
    run_sweep(sweep_file,execute=True)
    assert result.read_text()==before
