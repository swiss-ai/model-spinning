#!/bin/bash
#SBATCH --job-name={{job_name}}
#SBATCH --container-writable
#SBATCH --environment={{environment}}
#SBATCH --time={{time_limit}}
#SBATCH --ntasks-per-node=1
#SBATCH --dependency=singleton
#SBATCH --account=a-infra01
#SBATCH --output={{job_name}}-%j.out
#SBATCH --error={{job_name}}-%j.err

export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export PROMETHEUS_MULTIPROC_DIR=/ocfbin/scratch
export SP_NCCL_SO_PATH=/usr/lib/x86_64-linux-gnu/
export NO_PROXY=0.0.0.0,127.0.0.1

export MODEL_PATH={{model_path}}
export PARSER_ARGS="{{model_args}}"
export MODEL_NAME={{model_name}}

/ocfbin/ocf-v2 start --bootstrap.addr {{bootstrap_addr}} --subprocess "{{sub_process}}" --service.name llm --service.port 8080