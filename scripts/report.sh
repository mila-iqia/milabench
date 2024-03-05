


# results might be owned by root because of docker
chown -R $USER:$USER results

# install the right version of milabench
pip install git+https://github.com/mila-iqia/milabench@v0.0.8

# Download the AO config
wget https://gist.githubusercontent.com/Delaunay/ef0b2dc4aae5d42c16f00a17d17d490e/raw/acc5dabb908b3b9f918c9c5716711d4b82569da7/AO_476.yaml

# get the official score
milabench report --config AO_476.yaml --runs results/
