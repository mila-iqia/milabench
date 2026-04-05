github-login:
	echo $(GITHUB_MILABENCH_TOK) | sudo docker login ghcr.io -u Delaunay --password-stdin


docker-ngc: docker-ngc-build docker-ngc-push

docker-ngc-build:
	sudo docker build 														\
		--build-arg CACHEBUST=`git rev-parse $(git branch --show-current)`	\
	 	-f docker/Dockerfile-ngc 											\
		-t milabench:cuda-ngc-25.10 . 

	sudo docker tag 														\
		milabench:cuda-ngc-25.10											\
	 	ghcr.io/mila-iqia/milabench:cuda-ngc-25.10

docker-ngc-push:
	sudo docker push ghcr.io/mila-iqia/milabench:cuda-ngc-25.10

docker-build:
	sudo docker build 														\
		--progress=plain													\
		--build-arg CACHEBUST=`git rev-parse $(git branch --show-current)`	\
	 	-f docker/Dockerfile-cuda 											\
		-t milabench:cuda-cuda-nightly . 

	sudo docker tag 														\
		milabench:cuda-cuda-nightly											\
	 	ghcr.io/mila-iqia/milabench:cuda-cuda-nightly

	sudo docker push ghcr.io/mila-iqia/milabench:cuda-cuda-nightly

tests:
	coverage run --source=milabench -m pytest --ignore=tests/integration tests/ -vv -x 



# docker build                                                \
# 	-f docker/Dockerfile-cuda                 				\
# 	--build-arg CONFIG=all.yaml        				\
# 	--build-arg SELECT=whisper-transcribe-single           	\
# 	-t cuda-whisper-transcribe-single                 		\
# 	.

docker run --rm --gpus all  -it                                            \
	--ipc=host --ulimit memlock=-1 --ulimit stack=67108864              \
	--network=host                                                      \
	--security-opt=label=disable                                        \
	-v "/home/github/.ssh/id_ed25519_shared:/root/.ssh/id_rsa:Z"        \
	-e HF_TOKEN=<>                    									\
	-v "/opt/milabench/data:/milabench/results/data"                   \
	-v "/opt/milabench/cache:/milabench/results/cache"                 \
	-v "/opt/milabench/runs/manual:/milabench/results/runs" 			\
	"docker.io/library/cuda-whisper-transcribe-single"                  \
	bash
	
	milabench run                                                   	\
		--config /milabench/milabench/config/all.yaml                \
		--system /milabench/results/data/system.yaml \
		--select whisper-transcribe-single