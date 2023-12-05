
# Args
# ====
GPU_ID=0
MIG_CONFIG="19,19,19,19,19,19,19"

# ---



# nvidia-smi -L

list_mig_profile () {
    # List Mig profiles
    #
    #   lgip: List GPU Instance Profile
    #
    sudo nvidia-smi mig -lgip
}

enable_mig () {
    #
    # enable mig of a specific GPU, or a list of GPUs
    #

    # sudo systemctl stop nvsm
    # sudo systemctl stop dcgm

    sudo nvidia-smi -i ${GPU_ID} -mig 1

    # sudo nvidia-smi --gpu-reset
}

create_gpu_instance() {
    #
    #   cgi: Create GPU Instance
    #
    # Creating GPU instances
    #     one instance of (9)
    # and one instance of (3g.20gb)
    #
    sudo nvidia-smi mig -cgi ${MIG_CONFIG} -C
}


create_compute_instance () {
    #
    # Create Compute Instance !?
    #
    # sudo nvidia-smi mig -cci 0,0,0 -gi 1
}

list_gpu_instance () {
    # List gpu instance
    sudo nvidia-smi mig -lgi
}

destroy_gpu_compute_instance () {
    # destroy GPU instance
    sudo nvidia-smi mig -dci
    sudo nvidia-smi mig -dgi
}

