from src.entrypoint import dispatch
from src.training import main as legacy_main

if __name__ == "__main__":
    dispatch("train_main", legacy_main)
