



conda activate py310

rm -rf /tmp/workspace/runs
CUDA_VISIBLE_DEVICES=0 milabench run --system /tmp/workspace/system.yaml --base /tmp/workspace --config /home/mila/d/delaunap/scratch/shared/milabench/config/standard.yaml --select resnet50

