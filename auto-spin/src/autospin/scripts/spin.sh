#!/bin/bash
#SBATCH --job-name={{job_name}}
#SBATCH --container-writable
#SBATCH --time={{time_limit}}
#SBATCH --ntasks-per-node=1
#SBATCH --dependency=singleton
#SBATCH --partition normal
#SBATCH --account=a-infra01
#SBATCH --output={{job_name}}-%j.out
#SBATCH --error={{job_name}}-%j.err

unset HTTPS_PROXY HTTP_PROXY http_proxy https_proxy no_proxy

export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export PROMETHEUS_MULTIPROC_DIR=/ocfbin/scratch
export SP_NCCL_SO_PATH=/usr/lib/x86_64-linux-gnu/

export MODEL_PATH={{model_path}}
export PARSER_ARGS="{{model_args}}"
export MODEL_NAME={{model_name}}


{% if 'v0.1.1' == ocf_version %}
 srun -N ${SLURM_JOB_NUM_NODES} --environment={{environment}} --container-writable bash -c '\
    cd /tmp
    curl -L "https://github.com/ResearchComputer/OpenComputeFramework/releases/download/v0.1.1/ocf-amd64" > ocf-amd64
    chmod +x ocf-amd64
    ./ocf-amd64 start --bootstrap.addr {{bootstrap_addr}} --subprocess "{{sub_process}}" --service.name llm --service.port 8080
    '
{% else %}
 srun -N ${SLURM_JOB_NUM_NODES} --environment={{environment}}  --container-writable bash -c '\
    /ocfbin/ocf-v2 start --bootstrap.addr {{bootstrap_addr}} --subprocess "{{sub_process}}" --service.name llm --service.port 8080
    '
{% endif %}
