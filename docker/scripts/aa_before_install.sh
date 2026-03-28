echo "torchcodec"                                                   
apt-get update                                                      
apt-get install -y pybind11-dev pkg-config ffmpeg libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev libavutil-dev libswresample-dev libswscale-dev

I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 pip install "git+https://github.com/pytorch/torchcodec.git@release/0.7" --no-build-isolation         
