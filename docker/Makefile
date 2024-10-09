



hpu:
	git add --all
	git commit -m "-"
	git push origin hpu
	docker rmi $(docker images --filter "dangling=true" -q --no-trunc) | true
	# docker system prune -a -f
	# docker image prune -a -f
	docker build --build-arg CACHEBUST=`git rev-parse hpu` -f Dockerfile-hpu -t dockerfile-hpu . 
	docker run   -it   --runtime=habana   -e HABANA_VISIBLE_DEVICES=all   -e OMPI_MCA_btl_vader_single_copy_mechanism=none --shm-size 50G   --cap-add=sys_nice   --net=host   dockerfile-hpu:latest bash -c '. $$MILABENCH_VENV/bin/activate && milabench prepare --use-current-env --select lightning && milabench run --use-current-env --select lightning'

