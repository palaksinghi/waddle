#!/bin/bash
##################################################################################
# Open Duck Bipedal — FRESH straight-line walk (FlashSAC)
# From scratch (no resume), 100k env steps, tensorboard + terminal reward prints.
##################################################################################
# Usage:
#   bash scripts/run_duck_walk_fresh100k.sh            # foreground
#   nohup bash scripts/run_duck_walk_fresh100k.sh > /tmp/opencode/duck_fresh100k.log 2>&1 &  # background
##################################################################################

# --- 1. Training (fresh, 100k steps) ---
# PYTHONUNBUFFERED=1 => reward prints appear live in the log (no 4KB buffering).
PYTHONUNBUFFERED=1 uv run python -u train.py \
    --overrides logger_type=tensorboard \
    --overrides seed=0 \
    --overrides env=duck \
    --overrides env.env_name=open_duck_bipedal \
    --overrides num_env_steps=100_000 \
    --overrides num_train_envs=1 --overrides num_eval_envs=1 --overrides num_record_envs=1 \
    --overrides num_eval_episodes=10 --overrides num_record_episodes=1 \
    --overrides agent=flashSAC \
    --overrides agent.buffer_max_length=100_000 \
    --overrides agent.buffer_min_length=5_000 \
    --overrides agent.buffer_device_type=cpu \
    --overrides agent.sample_batch_size=512 \
    --overrides agent.use_amp=false \
    --overrides updates_per_interaction_step=1 \
    --overrides logging_per_interaction_step=1000 \
    --overrides evaluation_per_interaction_step=10000 \
    --overrides metrics_per_interaction_step=10000 \
    --overrides recording_per_interaction_step=100000 \
    --overrides group_name=test --overrides exp_name=duck_walk_fresh100k \
    --overrides save_checkpoint_per_interaction_step=10000 \
    --overrides save_buffer_per_interaction_step=null

# --- 2. Monitor (another terminal) ---
# tail -f /tmp/opencode/duck_fresh100k.log
# ps aux | grep train.py | grep -v grep

# --- 3. TensorBoard (another terminal) ---
# uv run tensorboard --logdir runs/test/duck_walk_fresh100k --port 6006
# then open http://localhost:6006

# --- 4. Watch policy live (needs display) ---
# uv run python view_duck_policy.py \
#     --overrides env=duck --overrides env.env_name=open_duck_bipedal \
#     --overrides agent=flashSAC \
#     --overrides agent_load_path='models/test/duck_walk_fresh100k/open_duck_bipedal/<seed-dir>/step<N>'

# --- 5. Save video ---
# uv run python make_video.py \
#     --overrides env=duck --overrides env.env_name=open_duck_bipedal \
#     --overrides agent_load_path='models/test/duck_walk_fresh100k/open_duck_bipedal/<seed-dir>/step<N>' \
#     --out walk_fresh100k.mp4
