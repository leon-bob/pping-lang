"""`python -m pping_lang` = `pping-lang <子命令>`(见 cli.main)。"""
import sys

from pping_lang.cli import main

if __name__ == "__main__":
    sys.exit(main())
