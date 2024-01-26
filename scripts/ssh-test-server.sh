
function start_sshd() {
    USER="tester"
    sudo docker pull kabirbaidhya/fakeserver
    sudo docker run -it --rm -p 2222:22                                    \
            -v "/tmp/milabench/tests/config:/etc/authorized_keys/$USER"    \
            -e SSH_USERS="$USER:1001:1001"                                 \
            --name=fakeserver kabirbaidhya/fakeserver
}


function ssh_connect() {
    ssh -oCheckHostIP=no -oStrictHostKeyChecking=no -oPasswordAuthentication=no \
        -i /tmp/milabench/tests/config/id_rsa                                   \
        -p 2222                                                                 \
        tester@localhost pip install -e /tmp/milabench
}


function milabench_remote_install() {
    milabench install --config /tmp/milabench/config/standard.yaml --system /tmp/milabench/tests/config/system.yaml 
}