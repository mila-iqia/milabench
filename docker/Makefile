


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






  docker run --rm --gpus all                                            \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864              \
    --network=host                                                      \
    --security-opt=label=disable                                        \
    -e MILABENCH_REPORT_LEAN=1                                          \
    -e BENCHMATE_OBSERVATION_COUNT=10                                   \
    -e BENCHMATE_POLL_INTERVAL=5                                        \
    -v "/home/delaunap/.ssh/id_ed25519_shared:/root/.ssh/id_rsa:Z"       \
    -v "/opt/milabench/data:/milabench/results/data"                   \
    -v "/opt/milabench/cache:/milabench/results/cache"                 \
    -v "/opt/milabench/runs:/milabench/results/runs" 					\
    "docker.io/library/milabench-ci:dev-diffusion"                      \
    milabench run                                                       \
      --config "/milabench/milabench/config/ci.yaml"                    \
      --system /milabench/results/data/system.yaml                      \
      --select "diffusion-nodes"                             	        \
      --use-current-env


  docker run -it --rm --gpus all                                        \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864              \
    --network=host                                                      \
    --security-opt=label=disable                                        \
    -e MILABENCH_REPORT_LEAN=1                                          \
    -e BENCHMATE_OBSERVATION_COUNT=10                                   \
    -e BENCHMATE_POLL_INTERVAL=5                                        \
    -v "/home/delaunap/.ssh/id_ed25519_shared:/root/.ssh/id_rsa:Z"      \
    -v "/opt/milabench/data:/milabench/results/data"                    \
    -v "/opt/milabench/cache:/milabench/results/cache"                  \
    -v "/opt/milabench/runs:/milabench/results/runs" 					\
    "docker.io/library/milabench-ci:dev-diffusion"                      \
    bash


docker build                                   	\
    -f "docker/Dockerfile-ngc"                	\
    --build-arg CONFIG=ci.yaml        			\
    --build-arg SELECT="dimenet,pna"           	\
    -t "milabench-ci:dev-geo_gnn"             	\
    .
