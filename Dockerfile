ARG FROM_IMAGE_NAME=nvcr.io/nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
FROM ${FROM_IMAGE_NAME}

# Install dependencies for system configuration logger
RUN apt-get update && apt-get install -y --no-install-recommends --allow-unauthenticated \
        infiniband-diags \
	git \
	vim \
	cgroup-bin \
	cgroup-lite \
        pciutils && \
    rm -rf /var/lib/apt/lists/*

# Clone MILA benchmarks
WORKDIR /workspace/milabench
COPY . .

# Install dependencies
RUN ./scripts/install-apt-packages.sh
RUN ./scripts/install_conda.sh --no-init

ENV PATH=/root/anaconda3/bin:$PATH

RUN conda create -n mlperf python=3.7 -y

SHELL ["conda", "run", "-n", "mlperf", "/bin/bash", "-c"]
RUN conda install poetry -y
RUN poetry install

# Configure environment variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Set the entry point to create the cgroups automatically
ENTRYPOINT ["scripts/docker_entry.sh"]
