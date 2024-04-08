# orbit_safe_rl

- omniverse version : 2023.1.1
- orbit version : release 0.2.0
- reference paper : https://arxiv.org/pdf/2303.04118.pdf

```Python
# 1) execute from the root directory of the repository
${work_dir}/rsl_rl_custom/train.py --task Isaac-Reach-obs-UR10-v0 --headless

# 2) execute from the root directory of the repository
${work_dir}/rsl_rl_custom/play.py --task Isaac-Reach-obs-UR10-Play-v0

# option) Tensorboard (execute from the root directory of the repository)
./orbit.sh -p -m tensorboard.main --logdir=logs  

```


### launch.json
``` json
      {
        "name": "isaac_24_1",
        "type": "debugpy",
        "request": "launch",
        "program": "${file}",
        "console": "integratedTerminal",
        "env": {
          "EXP_PATH": "${workspaceFolder}/_isaac_sim/apps",
          "RESOURCE_NAME": "IsaacSim"
        },
        "envFile": "${workspaceFolder}/.vscode/.python.env",
        "preLaunchTask": "setup_python_env",
        "args": [
          // "--task", "Isaac-Reach-Franka-v0",
          // "--task", "Isaac-Open-Drawer-Franka-v0",

          // "--task", "Isaac-Cartpole-v2",
          // "--task", "Isaac-Reach-obs-UR10-v0",
          "--task", "Isaac-Reach-obs-UR10-Play-v0",

          // ==== FOR TRAINING ====
          "--num_envs", "4",
          // "--headless",

          # ==== FOR EVALUATION ====
          // "--num_envs", "2",
          // "--load_run", "2024-04-04_10-10-56",  
        ],
        "justMyCode": false
      },
```
