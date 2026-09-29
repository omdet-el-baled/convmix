import torch
import pytest
from src.config import load_config


def pytest_sessionstart(session):
    torch.set_num_threads(1)


@pytest.fixture
def tiny_config(tmp_path):
    cfg = load_config('configs/base.yaml')
    cfg['data'].update(data_dir=str(tmp_path/'data'), train=8, val=4, test=4,
                       duration=0.016, sample_rate=2000, filter_length=5, batch_size=2,
                       num_workers=0, pin_memory=False, persistent_workers=False)
    cfg['network'].update(base_channels=8, time_dim=16, num_heads=2, dropout=0.05)
    cfg['sde']['steps'] = 32
    cfg['run'].update(name='tiny',output_dir=str(tmp_path/'runs'),tensorboard=False)
    cfg['trainer'].update(accelerator='cpu',max_epochs=1,limit_train_batches=2,
                          limit_val_batches=1,num_sanity_val_steps=0,log_every_n_steps=1)
    cfg['sampling'].update(num_steps=2)
    cfg['evaluation'].update(batch_size=2,output_dir=str(tmp_path/'results'))
    return cfg
