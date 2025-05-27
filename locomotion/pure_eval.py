import argparse
import os
import sys
import pickle
import glob
import re
import torch
import onnx

import genesis as gs

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rsl_rl.runners import OnPolicyRunner
from pure_env import PureEnv
from pure_train import get_cfgs
from pure_plot import PurePlot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="pure")
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("-p", "--enable_plot", type=bool, default=True)
    args = parser.parse_args()

    gs.init(logging_level="error")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    # reward_cfg["reward_scales"] = {}

    # visualize the target
    env_cfg["visualize_target"] = True
    # set the max FPS for visualization
    env_cfg["max_visualize_FPS"] = 60
    # set the episode length to 10 seconds
    env_cfg["episode_length_s"] = 1000.0

    env = PureEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    if args.ckpt is None:
        model_files = glob.glob(os.path.join(log_dir, "model_*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No model checkpoints found in {log_dir}")

        iterations = []
        for file in model_files:
            match = re.search(r"model_(\d+)\.pt", os.path.basename(file))
            if match:
                iterations.append(int(match.group(1)))

        if not iterations:
            raise ValueError(f"Could not parse iteration numbers from model files in {log_dir}")

        latest_ckpt = max(iterations)
        print(f"Using latest checkpoint: {latest_ckpt}")
    else:
        latest_ckpt = args.ckpt

    resume_path = os.path.join(log_dir, f"model_{latest_ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()

    # export to ONNX
    onnx_path = os.path.join(log_dir, f"policy_{latest_ckpt}.onnx")
    torch.onnx.export(
        runner.alg.actor_critic.actor,
        obs,  # Add batch dimension
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
    )

    reward_names = list(env.reward_functions.keys())
    sample_obs = env.get_observations()
    obs_dim = sample_obs.shape[1]

    if args.enable_plot:
        print("Plotting enabled")
        visualizer = PurePlot(reward_names, obs_dim)

    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)

            reward_values = {}
            for name, reward_func in env.reward_functions.items():
                raw_value = reward_func()[0].item()
                scale = reward_cfg["reward_scales"].get(name, 1.0)  # Default to 1.0 if no scale
                reward_values[name] = raw_value * scale

            if args.enable_plot:
                visualizer.update(reward_values, obs)


if __name__ == "__main__":
    main()
