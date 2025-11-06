


bench = rlhf-gpus
# bench = "lightning"
lazy = 0

hpu:
	git add --all
	git commit -m "-" | true
	git push origin hpu
	docker rmi -f $(docker images --filter "dangling=true" -q --no-trunc) | true
	# docker system prune -a -f
	# docker image prune -a -f
	docker build --build-arg BENCH=$(bench) --build-arg CACHEBUST=`git rev-parse hpu` -f Dockerfile-hpu -t dockerfile-hpu . 
	docker run --rm -it   --runtime=habana  -e PT_HPU_LAZY_MODE=$(lazy) -e HABANA_VISIBLE_DEVICES=all   -e OMPI_MCA_btl_vader_single_copy_mechanism=none --shm-size 50G   --cap-add=sys_nice   --net=host   dockerfile-hpu:latest bash -c '. $$MILABENCH_VENV/bin/activate && milabench install --use-current-env --select $(bench) && pip uninstall torch torchvision torchaudio -y  &&  milabench prepare --use-current-env --select $(bench) && milabench run --use-current-env $(args) --select $(bench)'


github-login:
	echo $(GITHUB_MILABENCH_TOK) | sudo docker login ghcr.io -u Delaunay --password-stdin

ngc:
	sudo docker build 															\
		--build-arg CACHEBUST=`git rev-parse $(git branch --show-current)`	\
	 	-f Dockerfile-ngc 													\
		-t milabench:cuda-ngc-nightly . 
	
	sudo docker tag 																\
		milabench:cuda-ngc-nightly											\
	 	ghcr.io/mila-iqia/milabench:cuda-ngc-nightly

	sudo docker push ghcr.io/mila-iqia/milabench:cuda-ngc-nightly