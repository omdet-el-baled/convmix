from src.entrypoint import dispatch
from src.evaluation import main as legacy_main

if __name__ == "__main__":
    dispatch("evaluate_main", legacy_main)
