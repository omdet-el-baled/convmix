import copy
import json
from pathlib import Path
import pytest
import torch

from src.config import load_config,set_value,config_hash
from src.data.generate import generate_dataset
from src.training import train
from src.evaluation import evaluate
from src.factory import load_model_from_checkpoint
from src.utils import load_checkpoint


def test_config_validation_and_inheritance():
    cfg=load_config('configs/base.yaml',['sde.alpha=1.0','trainer.max_epochs=10'])
    assert cfg['sde']['alpha']==1 and cfg['trainer']['max_epochs']==10
    with pytest.raises(KeyError):set_value(cfg,'sde.alhpa',3)
    assert config_hash(cfg)==config_hash(copy.deepcopy(cfg))


@pytest.mark.integration
def test_short_lightning_train_checkpoint_evaluate_resume(tiny_config):
    cfg=tiny_config
    generate_dataset(cfg)
    # Two physical training updates with accumulation=2 and 4 batches.
    cfg['trainer'].update(limit_train_batches=4,accumulate_grad_batches=2)
    run=train(cfg)
    checkpoint=run/'checkpoints/last.ckpt'
    assert checkpoint.exists()
    payload=load_checkpoint(checkpoint)
    assert payload['global_step']==2
    assert int(payload['state_dict']['ema_updates'])==2
    model,saved,meta=load_model_from_checkpoint(checkpoint)
    assert saved['sde']==cfg['sde']
    assert not model.training and not model.ema_score_model.training
    out,summary=evaluate(checkpoint,device='cpu')
    assert summary['num_examples']==4 and summary['mean_nfe']==2
    assert summary['reverse_final_time']==cfg['model']['t_eps']
    with pytest.raises(FileExistsError):evaluate(checkpoint,device='cpu')
    # Resume increases epoch budget; never silently treats a shorter checkpoint
    # as an already completed longer experiment.
    cfg2=copy.deepcopy(cfg);cfg2['trainer']['max_epochs']=2
    cfg2['run']['resume']=str(checkpoint)
    train(cfg2)
    newer=load_checkpoint(checkpoint)
    assert newer['global_step']==4 and int(newer['state_dict']['ema_updates'])==4
