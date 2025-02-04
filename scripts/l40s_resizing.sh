



conda activate py310

rm -rf /tmp/workspace/runs
# CUDA_VISIBLE_DEVICES=0 
export MILABENCH_SIZER_CONFIG=/home/mila/d/delaunap/scratch/shared/milabench/config/scaling/L40S.yaml
milabench run --system /tmp/workspace/system.yaml --base /tmp/workspace --config /home/mila/d/delaunap/scratch/shared/milabench/config/standard.yaml --exclude nobatch > out.txt
