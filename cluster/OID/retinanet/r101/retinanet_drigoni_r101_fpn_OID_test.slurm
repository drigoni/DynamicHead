#!/bin/bash
### Owner: Davide Rigoni

### ============ SLURM COMMANDS
#SBATCH --job-name='retinanet_drigoni_r101_fpn_OID_test'     # create a short name for your job
#SBATCH --mail-user=davide.rigoni.2@phd.unipd.it
#SBATCH --output=cluster/out/retinanet/r101/retinanet_drigoni_r101_fpn_OID_test_dataset-%a.out
#SBATCH --partition=gpu
#SBATCH --mem=150G				                    # total memory per node
#SBATCH --time=2-00:00:00                           # total run time limit (DD-HH:MM:SS)
#SBATCH --nodes=1                                   # node count
#SBATCH --ntasks-per-node=1                         # total number of tasks per node
#SBATCH --cpus-per-task=64		                    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-node=1                           # number of gpus per node
### SBATCH --exclusive                                 # for exclusive access
#SBATCH --dependency=afterany:47082062

#SBATCH --array=0-2
#---- MORE COMMANDS
#----#SBATCH --mail-type=begin                           # send email when job begins
#--- #SBATCH --mail-type=end                             # send email when job ends
#--- #SBATCH --error=cluster/err/dh_train.err
#--- #SBATCH --ntasks=2
#--- #SBATCH --gpus-per-task=4
#--- #SBATCH --constraint=A5000
#--- #SBATCH --gres=gpu:4                                # number of gpus per node


### ============ VARIABLES SETTING
SLURM_MASTER_PORT=$(expr 10000 + $(echo -n ${SLURM_JOBID} | tail -c 4))
SLURM_MASTER_NODE="$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)" # such as gn11
SLURM_MASTER_ADDR="${SLURM_MASTER_NODE}"
SLURM_MASTER_URL="tcp://${SLURM_MASTER_ADDR}:${SLURM_MASTER_PORT}"
export SLURM_MASTER_PORT=$SLURM_MASTER_PORT
export SLURM_MASTER_NODE=$SLURM_MASTER_NODE
export SLURM_MASTER_ADDR=$SLURM_MASTER_ADDR

# model variables
NUM_GPUS=${SLURM_GPUS_PER_NODE}
NUM_MACHINES=${SLURM_NNODES}
BATCH_SIZE=$(expr 5 \* $(echo -n ${SLURM_JOB_NUM_NODES}) \* $(echo -n ${SLURM_GPUS_PER_NODE})) # max 5 per A100
MAX_ITER=$(expr 32 \* 90000 \/ $(echo -n ${BATCH_SIZE}))

### ============ SOME PRINT COMMANDS
echo ''
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo 'Date: ' $(date)
echo 'Directory: ' $(pwd)
echo "Nodelist: " ${SLURM_JOB_NODELIST}                  # like gn11 or gn[37,58]
echo "Number of nodes: " ${SLURM_JOB_NUM_NODES}
echo "Ntasks per node: "  ${SLURM_NTASKS_PER_NODE}
echo "NGPUs per node: "  ${SLURM_GPUS_PER_NODE}
echo "CUDA_VISIBLE_DEVICES: " ${CUDA_VISIBLE_DEVICES}
echo "TORCH_DEVICE_COUNT: " $(python -c 'import torch; print(torch.cuda.device_count())')
echo "SLURM_MASTER_PORT: " ${SLURM_MASTER_PORT}
echo "SLURM_MASTER_NODE: " ${SLURM_MASTER_NODE}
echo "SLURM_MASTER_ADDR: " ${SLURM_MASTER_ADDR}
echo "SLURM_MASTER_URL: " ${SLURM_MASTER_URL}
echo "--------------------------------------------- "
echo "MODEL_NUM_GPUS: " ${NUM_GPUS}
echo "MODEL_NUM_MACHINES: " ${NUM_MACHINES}
echo "MODEL_BATCH_SIZE: " ${BATCH_SIZE}
echo "MODEL_MAX_ITER: " ${MAX_ITER}
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo ''


# model config
CONFIG_FILE=./configs/OID/retinanet/r101/retinanet_drigoni_r101_fpn_OID_test.yaml
DATSETS=('("oid_v4_val",)' '("oid_v4_val_subset",)' '("oid_v4_val_powerset",)')
srun python train_net.py \
                    --config  ${CONFIG_FILE} \
                    --num-gpus ${NUM_GPUS} \
                    --num-machines ${NUM_MACHINES} \
                    --dist-url ${SLURM_MASTER_URL} \
                    --eval-only \
                    SOLVER.IMS_PER_BATCH ${BATCH_SIZE} \
                    SOLVER.MAX_ITER ${MAX_ITER} \
                    DATASETS.TEST ${DATSETS[$SLURM_ARRAY_TASK_ID]}



# last default print
echo ''
echo ''
echo 'Job done.'
echo 'Date: ' $(date)