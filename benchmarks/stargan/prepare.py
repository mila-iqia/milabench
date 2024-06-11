


def download_celebA():
    from datasets import load_dataset
    dataset = load_dataset(
        "student/celebA", 
        revision="2d31e6555722815c74ea7c845b07c1063dd705e9",
        cache_dir="/tmp/milabench/cuda/results/data"
    )


if __name__ == "__main__":
    download_celebA()
