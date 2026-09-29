# Compatibility entry point after `python -m pip install -e . --no-deps`.
from src.data.generate import main
if __name__ == "__main__":
    main()
