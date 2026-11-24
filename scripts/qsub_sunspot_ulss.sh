#!/bin/bash -x
#PBS -l select=8
#PBS -l place=scatter
#PBS -l walltime=00:20:00
#PBS -q workq
#PBS -A datascience
#PBS -l filesystems=home:tegu
#PBS -k doe
#PBS -e /lus/tegu/projects/datasets/software/testing/logs_alcf4_aicomm/errordir
#PBS -o /lus/tegu/projects/datasets/software/testing/logs_alcf4_aicomm/outdir
#PBS -j oe
#PBS -N AICOMM_ULSS12

export TZ='/usr/share/zoneinfo/US/Central'

WORK_DIR=/lus/tegu/projects/datasets/software/testing/alcf4-aicomm
LOG_ROOT=/lus/tegu/projects/datasets/software/testing/logs_alcf4_aicomm
LOG_WRAPPER=${WORK_DIR}/scripts/log_wrapper.sh

SEQ=16896
HID=25872
TIMING_LOOPS=8
WARMUP=3
PRECISION="bfloat16"
N_LAYERS=1
IN_TYPE="random"
BUCKET=1e9
TRIAL=1
TP_DEGREE=12

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=12
let NRANKS=${NNODES}*${NRANKS_PER_NODE}
export PALS_WORLD_SIZE=${NRANKS}  # PALS does not set this

module load frameworks

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_RX_MATCH_MODE=hybrid
export CCL_PROCESS_LAUNCHER=pmix
#export CCL_KVS_MODE=mpi  # for >1024 nodes

## PPN=1
#export CPU_AFFINITY="list:4-7"
#export CCL_WORKER_AFFINITY="42"
#export ZE_AFFINITY_MASK="0"

## PPN=12
export CPU_AFFINITY="list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79"
export CCL_WORKER_AFFINITY="42,43,44,45,46,47,94,95,96,97,98,99"
#export ZE_AFFINITY_MASK="0,1,2,3,4,5,6,7,8,9,10,11"  # PALS rejects tile indices >5

#echo "========= ENVIRONMENT VARIABLES ======="
#env
#echo "========= ENVIRONMENT VARIABLES ======="

RUN_ID=sunspot_ULSS_SEQ${SEQ}_HID${HID}_TP${TP_DEGREE}_L${N_LAYERS}_TL${TIMING_LOOPS}_${PRECISION}_${IN_TYPE}_ELEM_${BUCKET}_N${NNODES}_R${NRANKS_PER_NODE}_T${TRIAL}_$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR=${LOG_ROOT}/logs/${RUN_ID}
mkdir -p ${LOG_DIR}
echo "${RUN_ID}"

# --pmi=pmix: log_wrapper.sh reads PMIX_RANK
mpiexec --pmi=pmix -n ${NRANKS} -ppn ${NRANKS_PER_NODE} -l --line-buffer --cpu-bind ${CPU_AFFINITY} \
    -env MASTER_ADDR=$(hostname).hsn.cm.sunspot.alcf.anl.gov -env MASTER_PORT=2345 \
    ${LOG_WRAPPER} python ${WORK_DIR}/src/aicomm_bench.py -dvc "xpu" \
    -tp_degree=${TP_DEGREE} --sequence_length=${SEQ} --hidden_dimension=${HID} --barrier -ulysses_enable \
    --iterations=${TIMING_LOOPS} --warmup_iterations=${WARMUP} --precision ${PRECISION} -n_layers ${N_LAYERS} -bucket ${BUCKET} \
    --logging --log_directory=${LOG_DIR} --log_file=${RUN_ID}.log
