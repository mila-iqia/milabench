# This is the script run by milabench run (by default)

# It is possible to use a script from a GitHub repo if it is cloned using
# clone_subtree in the benchfile.py, in which case this file can simply
# be deleted.

import time

import voir
from giving import give


def main():
    for i in voir.iterate("train", range(10000), report_batch=True, batch_size=64):
        give(loss=1 / (i + 1))
        time.sleep(0.1)


if __name__ == "__main__":
    main()
