mpirun --allow-run-as-root -np 8 \
--mca pml ucx \
--mca btl ^openib \
-x NCCL_DEBUG=VERSION \
/workspace/rccl-tests/build/all_reduce_perf \
-b 1 \
-e 16G \
-f 2 \
-g 1