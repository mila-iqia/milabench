from torchvision.datasets import MNIST
import sys

if __name__ == "__main__":
    dest = sys.argv[1]
    print(f"Downloading MNIST into {dest}/MNIST...")
    MNIST(dest, download=True)
    print("Done!")
