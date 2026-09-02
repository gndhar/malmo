#!/bin/bash
#PBS -N train_phasenet
#PBS -q gpuq
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=04:00:00
#PBS -o train_out.log
#PBS -e train_err.log
#PBS -k oe
cd $PBS_O_WORKDIR
module load gcc12.3.0
module load cuda12.4
source .venv/bin/activate
export OMP_NUM_THREADS=$NCPUS
# malmo/ml should contain data_gen.py, zern.py, forward.py, rm.py,
# dual_branch_phasenet.py, and this train.py.
cd $PBS_O_WORKDIR
# --- start GPU utilization logging in the background ---
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv -l 5 > gpu_util.csv &
MONITOR_PID=$!
# Fine-tuning from checkpoint/grf_1/best.pth (weights only — fresh
# optimizer/scheduler/epoch counter, since we're switching to a noise-
# augmented training regime rather than resuming the grf_1 run) with
# random Rk noise injection enabled for sim-to-real robustness.
python train.py \
    --N 32 \
    --zern_n 8 \
    --num_workers 8 \
    --epochs 200 \
    --batch_size 64 \
    --lr 1e-3 \
    --scheduler plateau \
    --plateau_factor 0.5 \
    --plateau_patience 5 \
    --eta_min 1e-7 \
    --load_ckpt_path ./checkpoint/grf_1/best.pth \
    --weights_only \
    --train_size 4096 \
    --val_size 1024 \
    --checkpoint_dir ~/malmo/ml/checkpoint \
    --cache_dir ~/malmo/ml/cache \
    --log_dir ~/malmo/ml/runs \
    --run_name grf_1_noise \
    --train_noise \
    --snr_min -20 \
    --snr_max 30 \
    --clean_prob 0.1 \
    --noise_warmup_epochs 10 \
    --val_snr_bins "30,20,10,0,-10,-20" \
    --val_snr_every 5
# --- stop GPU logging once training is done ---
kill $MONITOR_PID
