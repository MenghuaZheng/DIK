docker run -it  \
--name amd_dik_vllm \
--network=host \
--group-add=video \
--ipc=host \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
--privileged \
--device /dev/kfd \
--device /dev/dri \
-v /home/zhengmenghua:/home/zhengmenghua \
-v /mnt/:/app/models \
rocm/vllm:latest