
OUTPUT="test.out"
rm -rf $OUTPUT
touch $OUTPUT
sbatch  --ntasks=1\
        --gpus-per-task=rtx8000:1\
        --cpus-per-task=4\
        --time=01:30:00\
        --ntasks-per-node=1\
        --mem=64G\
        -o $OUTPUT\
        slurm.sh\
        -a cuda\
        -b stable

tail -f $OUTPUT