import subprocess
import ray
from ray.tune.registry import register_env
from numpy.random import default_rng

ray.init(num_cpus=96, num_gpus=8)

num_gpus = 8
num_cpus = 96
exp_per_cpus = 4
num_exps = num_cpus / exp_per_cpus
gpu_fractions = num_gpus / num_exps

rng = default_rng()
seeds = rng.choice(100, size=int(num_exps), replace=False)
print(seeds)

# Scripts
#python train_rl2.py --env_name HalfCheetahDirEnv --device "cuda:0" --config_path ./configs/cheetah_dir_config.yaml --seed 1006

cmds = []
for seed in seeds:
    cmd = "python train_ppo_clean_rl2.py --meta_learning False --shuffle_mb True --same_task True --env_name HalfCheetah-v3 --device 'cuda:0' --config_path ./configs/cheetah_dir_config.yaml --seed {}".format(seed)
    cmds.append(cmd)

@ray.remote(num_cpus=exp_per_cpus, num_gpus=gpu_fractions)
def distributed_train(command):
    subprocess.run(command, shell=True)

refs = [distributed_train.remote(cmd) for cmd in cmds]
results = ray.get(refs)

ray.shutdown()