#! /bin/bash
set +e
set -x

x=1
while [ $x -gt 0 ]
do
  mpirun --allow-run-as-root --mca coll_tuned_use_dynamic_rules 1 --mca coll_tuned_allgatherv_algorithm 3 \
    --mca coll_tuned_allgather_algorithm 4  --mca btl_openib_cuda_async_recv false --mca btl_openib_rroce_enable 1  \
    --mca btl_openib_want_cuda_gdr 0 --mca btl_openib_cpc_include rdmacm --mca mpi_warn_on_fork false -bind-to none \
    -map-by slot -x NCCL_DEBUG=INFO  -x NCCL_CHECKS_DISABLE=1 -x NCCL_SHM_DISABLE=0 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 \
    -mca btl ^openib -mca btl_tcp_if_include eth0 sh mpi_pretrain_core.sh
  sleep 60
done
