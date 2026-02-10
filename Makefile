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
		--build-arg CACHEBUST=`git rev-parse $(git branch --show-current)`	\
	 	-f docker/Dockerfile-cuda 											\
		-t milabench:cuda-cuda-nightly . 

	sudo docker tag 														\
		milabench:cuda-cuda-nightly											\
	 	ghcr.io/mila-iqia/milabench:cuda-cuda-nightly
