


bench = rlhf-gpus
lazy = 0
token = ""
compile = 1
devices = "all"

# 
hpu:
	git add --all
	git commit -m "-" | true
	git push origin gaudi3
	sudo docker rmi -f $(docker images --filter "dangling=true" -q --no-trunc) | true
	# docker system prune -a -f
	# docker image prune -a -f
	sudo docker build --build-arg BENCH=$(bench) --build-arg CACHEBUST=`git rev-parse gaudi3` -f Dockerfile-hpu -t dockerfile-hpu . 
	sudo docker run --rm --runtime=habana 								    	\
		-e TORCHCOMPAT_COMPILE=$(compile)								      	\
		-e PT_HPU_LAZY_MODE=$(lazy)												\
		-e MILABENCH_HF_TOKEN=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  			\
		-e HABANA_VISIBLE_DEVICES=$(devices)   					     			\
		-e OMPI_MCA_btl_vader_single_copy_mechanism=none		    			\
		 --ipc=host   --cap-add=sys_nice   --net=host   dockerfile-hpu:latest   \
		 bash -c '. $$MILABENCH_VENV/bin/activate && milabench prepare --use-current-env --select $(bench) && milabench run --use-current-env $(args) --select $(bench)'


hpu-fast:
	git add --all
	git commit -m "-" | true
	git push origin gaudi3
	sudo docker rmi -f $(docker images --filter "dangling=true" -q --no-trunc) | true
	# docker system prune -a -f
	# docker image prune -a -f
	sudo docker build --build-arg BENCH=$(bench) --build-arg CACHEBUST=`git rev-parse gaudi3` -f Dockerfile-hpu -t dockerfile-hpu . 
	sudo docker run --rm --runtime=habana 								    	\
		-e TORCHCOMPAT_COMPILE=$(compile)								      	\
		-e PT_HPU_LAZY_MODE=$(lazy)												\
		-e MILABENCH_HF_TOKEN=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  			\
		-e HABANA_VISIBLE_DEVICES=$(devices)   					     			\
		-e OMPI_MCA_btl_vader_single_copy_mechanism=none		    			\
		-v /home/ubuntu/hpu/results/data:/workspace/hpu/results/data 			\
		-v /home/ubuntu/hpu/results/cache:/workspace/hpu/results/data 			\
		 --ipc=host   --cap-add=sys_nice   --net=host   dockerfile-hpu:latest   \
		 bash -c '. $$MILABENCH_VENV/bin/activate && milabench run --use-current-env $(args) --select $(bench)'

