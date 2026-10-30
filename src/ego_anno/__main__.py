"""Run the ego-anno command-line interface with python -m ego_anno."""
import sys

from .cli import main


if __name__ == "__main__":
    sys.exit(main())
