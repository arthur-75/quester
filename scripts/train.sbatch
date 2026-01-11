#!/bin/bash
#SBATCH --job-name=q8B
#SBATCH --output=sbatch/logs/%x/slurm-%x-%j.out   # stdout log (%x=jobname, %j=jobid)
#SBATCH --error=sbatch/logs/%x/slurm-%x-%j.err    # stderr log
#SBATCH --nodes=1                      # single node
#SBATCH --gpus-per-node=1             
#SBATCH --time=48:00:00                # walltime limit
#SBATCH --partition=hard
#SBATCH --constraint=A6000

mkdir -p sbatch/logs/

# load conda
conda activate deepsql

# go to project root (if you run sbatch from elsewhere)
cd $SLURM_SUBMIT_DIR

# run training 
bash scripts/run.sh configs/${SLURM_JOB_NAME}.yaml
