export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_DEBUG=WARN          # 想看详细日志就留着
export NCCL_DMABUF_ENABLE=1
python test_rccl.py