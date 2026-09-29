"""One-example CLI and compatibility imports. Algorithms live in src/sampling."""
from src.data.filters import load_filters
from src.metrics import si_sdr, nmse
from src.sampling.samplers import (sample_euler, sample_heun, sample_pc,
    initialize_terminal_state, apply_drift, apply_diffusion_matrix,
    apply_diffusion_covariance, diffusion_g2)
from src.factory import load_model_from_checkpoint
from src.evaluation import main

if __name__ == "__main__":
    from src.entrypoint import dispatch
    dispatch("separate_main", lambda: main(single=True))
