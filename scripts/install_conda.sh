 
#!/bin/bash

set -ex

SCRIPT_PATH=$(dirname "$0")
TEMP_DIRECTORY=/tmp

if [ ! -f $HOME/anaconda3/bin/conda ]; then
    echo 'Install miniconda'
    wget -nc https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -P $TEMP_DIRECTORY

    chmod +x $TEMP_DIRECTORY/Miniconda3-latest-Linux-x86_64.sh 

    $TEMP_DIRECTORY/Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/anaconda3

    if [ $1 != --no-init ]; then
        $HOME/anaconda3/bin/conda init bash
    fi

    # $HOME/anaconda3/bin/conda create -n mlperf python=3.7 -y

    echo 'DONE: install miniconda'
fi
