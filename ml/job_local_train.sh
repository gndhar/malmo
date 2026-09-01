source .venv/bin/activate
python train.py \
    --N 8 \
    --zern_n 7 \
    --num_workers 8 \
    --epochs 2 \
    --batch_size 2 \
    --lr 4e-4 \
    --scheduler plateau \
    --plateau_factor 0.5 \
    --plateau_patience 5 \
    --eta_min 1e-6 \
    --load_ckpt none \
    --train_size 4 \
    --val_size 4 \
    --checkpoint_dir /tmp/checkpoint \
    --cache_dir /tmp/cache \
    --log_dir /tmp/runs
