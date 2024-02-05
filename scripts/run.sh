
OUTPUT="barebone.out"

rm -rf $OUTPUT
touch $OUTPUT

sbatch  --partition=staff-idt\
        --ntasks=1\
        --gpus-per-task=a100l:2\
        --cpus-per-task=4\
        --time=01:30:00\
        --ntasks-per-node=1\
        --mem=64G\
        -o $OUTPUT\
        slurm_barebone.sh\
        -a cuda\
        -b overhead

tail -f $OUTPUT
