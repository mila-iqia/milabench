import argparse
import json

from torchvision.datasets import MNIST

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Prepare the benchmark")
    parser.add_argument("--bench-config", help="Benchmark configuration")
    args = parser.parse_args()
    config = json.loads(args.bench_config)
    dest = config["dirs"]["data"]
    print(f"Downloading MNIST into {dest}/MNIST...")
    MNIST(dest, download=True)
    print("Done!")
