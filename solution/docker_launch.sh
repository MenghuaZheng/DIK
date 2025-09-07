docker run -it \
--name amd_dik_r62 \
--network=host \
--device=/dev/kfd \
--device=/dev/dri \
--group-add=video \
--ipc=host \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
--shm-size 32G \
--privileged \
--pid=host \
-v /home/zhengmenghua:/home/zhengmenghua \
-v $HOME/dockerx:/dockerx \
-w /home/zhengmenghua \
rocm/pytorch:rocm6.2.2_ubuntu22.04_py3.10_pytorch_release_2.1.2