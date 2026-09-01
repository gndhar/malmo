#!/bin/bash
#PBS -N train_phasenet_resume
#PBS -q gpuq
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=04:00:00
#PBS -o train_out.log
#PBS -e train_err.log
cd $PBS_O_WORKDIR
module load gcc12.3.0
module load cuda12.4
source .venv/bin/activate
export OMP_NUM_THREADS=$NCPUS
# malmo/ml should contain data_gen.py, zern.py, forward.py, rm.py,
# dual_branch_phasenet.py, and this train.py.
cd $PBS_O_WORKDIR/malmo/ml
# --- start GPU utilization logging in the background ---
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv -l 5 > gpu_util.csv &
MONITOR_PID=$!
# --load_ckpt options: "none" (scratch), "latest" (resume), "best" (fine-tune)
# Resuming: same --zern_n (5) as before, so this reads
# dual_branch_phasenet_latest_zern5.pth -- the checkpoint written by the
# --save_every 10 run. It restores model + optimizer + scheduler (including
# the plateau scheduler's patience counter) and picks up at epoch 100.
# --epochs is the *total* target, not "epochs to add" -- 200 here means
# "train up to epoch 200", i.e. 100 more on top of the 100 already done.
python train.py \
    --N 32 \
    --zern_n 5 \
    --num_workers 8 \
    --epochs 200 \
    --batch_size 32 \
    --lr 1e-3 \
    --save_every 10 \
    --scheduler plateau \
    --plateau_factor 0.5 \
    --plateau_patience 5 \
    --eta_min 1e-6 \
    --load_ckpt latest \
    --train_size 4096 \
    --val_size 1024 \
    --checkpoint_dir ~/malmo/ml/checkpoint \
    --cache_dir ~/malmo/ml/cache \
    --log_dir ~/malmo/ml/runs
# --- stop GPU logging once training is done ---
kill $MONITOR_PID
