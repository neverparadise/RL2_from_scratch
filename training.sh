#!/bin/bash

# Script to reproduce results

for ((i=0;i<10;i+=1))
do 
    python train.py \
    --exp_name "RL2_PPO" \
    --meta_learning True \ 
    --env_name "HalfCheetahDirEnv" \
    --device "cuda:0" \
    --seed $i
    --config_path ./configs/dir_config.yaml
    --save_periods 20 \
    --total-timesteps 1000000 \
    --rollout_steps 512 \
    --max_episode_steps 500 \
    --num-envs 1 \
done

