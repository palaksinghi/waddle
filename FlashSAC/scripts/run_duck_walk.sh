#!/bin/bash
##################################################################################
# Open Duck Bipedal — straight-line walk (FlashSAC, CPU sim)
# Resume from smoke-test step20000, train to 1M steps, log to wandb.
##################################################################################
# Usage:
#   bash scripts/run_duck_walk.sh            # foreground training
#   nohup bash scripts/run_duck_walk.sh > /tmp/opencode/duck_train.log 2>&1 &  # background
##################################################################################

# --- 1. Training (resume) ---
uv run python train.py \
    --overrides logger_type=tensorboard \
    --overrides seed=0 \
    --overrides env=duck \
    --overrides env.env_name=open_duck_bipedal \
    --overrides num_env_steps=1_000_000 \
    --overrides num_train_envs=1 --overrides num_eval_envs=1 --overrides num_record_envs=1 \
    --overrides num_eval_episodes=10 --overrides num_record_episodes=1 \
    --overrides agent=flashSAC \
    --overrides agent.buffer_max_length=1_000_000 \
    --overrides agent.buffer_min_length=10_000 \
    --overrides agent.buffer_device_type=cpu \
    --overrides agent.sample_batch_size=512 \
    --overrides agent.use_amp=false \
    --overrides updates_per_interaction_step=1 \
    --overrides group_name=test --overrides exp_name=duck_walk_straight \
    --overrides save_checkpoint_per_interaction_step=10000 \
    --overrides save_buffer_per_interaction_step=10000 \
    --overrides agent_load_path=models/test/duck_smoke_test/open_duck_bipedal/seed0-0926-005459/step20000

# --- 2. Monitor (run in another terminal) ---
# tail -f /tmp/opencode/duck_train.log
# ps aux | grep train.py | grep -v grep
# Wandb: https://wandb.ai/palak-vjti/FlashRL

# --- 3. Watch policy live (needs display) ---
# uv run python view_duck_policy.py \
#     --overrides agent_load_path='models/test/duck_walk_straight/open_duck_bipedal/<seed-dir>/step<N>'

# --- 4. Save video ---
# uv run python make_video.py \
#     --overrides env=duck --overrides env.env_name=open_duck_bipedal \
#     --overrides agent_load_path='models/test/duck_walk_straight/open_duck_bipedal/<seed-dir>/step<N>' \
#     --out walk_straight.mp4

# --- 5. Fresh training from scratch (no resume) ---
# Same as above but DROP the --overrides agent_load_path=... line.
