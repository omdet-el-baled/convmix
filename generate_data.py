from src.entrypoint import dispatch
from src.data.generate import main as legacy_main

if __name__ == "__main__":
    dispatch("generate_main", legacy_main)
