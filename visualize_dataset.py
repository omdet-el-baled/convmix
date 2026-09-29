from src.entrypoint import dispatch
from src.visualization import main as legacy_main

if __name__ == "__main__":
    dispatch("visualize_main", legacy_main)
