# Pull the image we are going to run
sudo docker pull ghcr.io/mila-iqia/milabench:cuda-nightly

# Run milabench
sudo docker run -it --rm --shm-size=8G                \
      --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all  \
      -v $(pwd)/results:/milabench/envs/runs          \
      ghcr.io/mila-iqia/milabench:cuda-nightly        \
      milabench run

# Show Performance Report
sudo docker run -it --rm --shm-size=8G                \
      --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all  \
      -v $(pwd)/results:/milabench/envs/runs          \
      ghcr.io/mila-iqia/milabench:cuda-nightly        \
      milabench summary /milabench/envs/runs

# Done
