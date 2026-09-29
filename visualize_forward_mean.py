import sys
from src.entrypoint import dispatch
from src.diagnostics import main

def legacy_main():
    sys.argv.insert(1, "mean")
    main()

if __name__ == "__main__":
    dispatch("mean_main", legacy_main)
