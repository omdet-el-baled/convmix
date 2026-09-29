from src.entrypoint import dispatch
from src.diagnostics import main as legacy_main

if __name__ == "__main__":
    dispatch("diagnose_main", legacy_main, legacy_extra=("mean", "score", "covariance"))
