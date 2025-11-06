github-login:
	echo $(GITHUB_MILABENCH_TOK) | sudo docker login ghcr.io -u Delaunay --password-stdin

docker-ngc:
	sudo docker build 														\
		--build-arg CACHEBUST=`git rev-parse $(git branch --show-current)`	\
	 	-f docker/Dockerfile-ngc 											\
		-t milabench:cuda-ngc-nightly . 
	
	sudo docker tag 														\
		milabench:cuda-ngc-nightly											\
	 	ghcr.io/mila-iqia/milabench:cuda-ngc-nightly

	sudo docker push ghcr.io/mila-iqia/milabench:cuda-ngc-nightly
