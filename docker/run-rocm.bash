
sudo docker run -it  --rm --shm-size=1G                        \
        --device=/dev/kfd --device=/dev/dri                      \
        --security-opt seccomp=unconfined --group-add video      \
        -v $(pwd)/results:/milabench/envs/runs                   \
        ghcr.io/mila-iqia/milabench:rocm-nightly                 \
        milabench run

sudo docker run -it  --rm --shm-size=1G                        \
        --device=/dev/kfd --device=/dev/dri                      \
        --security-opt seccomp=unconfined --group-add video      \
        -v $(pwd)/results:/milabench/envs/runs                   \
        ghcr.io/mila-iqia/milabench:rocm-nightly                 \
        milabench summary /milabench/envs/runs
