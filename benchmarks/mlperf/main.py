
import sys
import os


FOLDER = os.path.dirname(__file__)
BENCH = "training_results_v2.1/NVIDIA/benchmarks/bert/implementations/pytorch-preview"

print(sys.path)
sys.path.append(os.path.join(FOLDER, BENCH))
print(sys.path)

import run_squad
