#!/bin/bash
##################################################################################
# Open Duck Bipedal — STRAIGHT-LINE WALK v2 (FlashSAC)
# Resume from fresh100k step90000 (balances full episode, marches in place).
# New shaping: tight velocity tracking (std 0.15), forward x20, alive 0.25,
# stagnation truncation at 8 s -> standing no longer pays. 400k more steps.
##################################################################################
#   bash scripts/run_duck_walk_v2.sh
#   nohup bash scripts/run_duck_walk_v2.sh > /tmp/opencode/duck_v2.log 2>&1 &
##################################################################################

PYTHONUNBUFFERED=1 uv run python -u train.py \
    --overrides logger_type=tensorboard \
    --overrides seed=0 \
    --overrides env=duck \
    --overrides env.env_name=open_duck_bipedal \
    --overrides num_env_steps=400_000 \
    --overrides num_train_envs=1 --overrides num_eval_envs=1 --overrides num_record_envs=1 \
    --overrides num_eval_episodes=10 --overrides num_record_episodes=1 \
    --overrides agent=flashSAC \
    --overrides agent.buffer_max_length=400_000 \
    --overrides agent.buffer_min_length=5_000 \
    --overrides agent.buffer_device_type=cpu \
    --overrides agent.sample_batch_size=512 \
    --overrides agent.use_amp=false \
    --overrides updates_per_interaction_step=1 \
    --overrides logging_per_interaction_step=1000 \
    --overrides evaluation_per_interaction_step=10000 \
    --overrides metrics_per_interaction_step=10000 \
    --overrides recording_per_interaction_step=100000 \
    --overrides group_name=test --overrides exp_name=duck_walk_straight_v2 \
    --overrides save_checkpoint_per_interaction_step=10000 \
    --overrides save_buffer_per_interaction_step=null \
    --overrides agent_load_path=models/test/duck_walk_fresh100k/open_duck_bipedal/seed0-0926-173138/step90000

# Monitor: tail -f /tmp/opencode/duck_v2.log
# Board:   uv run tensorboard --logdir runs/test/duck_walk_straight_v2 --port 6007
