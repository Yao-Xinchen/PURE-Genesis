import argparse
import os
import sys
import pickle
import glob
import re
import torch
import threading
import time
from inputs import get_gamepad

import genesis as gs

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rsl_rl.runners import OnPolicyRunner
from pure_env import PureEnv


class GamepadController:
    def __init__(self):
        self.running = True
        self.lin_sensitivity = 0.75
        self.commands = torch.tensor([0.0, 0.0, 0.0], device="cuda:0")

        self.update_thread = threading.Thread(target=self.update_loop)

        self.update_thread.daemon = True
        self.update_thread.start()

    def stop(self):
        self.running = False
        self.update_thread.join()

    def update_loop(self):
        while self.running:
            try:
                events = get_gamepad()
                for event in events:
                    if event.code == "ABS_Y":
                        normalized_state = event.state / 128 - 1
                        self.commands[0] = normalized_state * self.lin_sensitivity
                    elif event.code == "ABS_X":
                        normalized_state = event.state / 128 - 1
                        self.commands[1] = normalized_state * self.lin_sensitivity
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

    def get_commands(self):
        return self.commands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="pure")
    parser.add_argument("--ckpt", type=int, default=None)

    args = parser.parse_args()

    gs.init(logging_level="error")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))

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

    controller = GamepadController()

    with torch.no_grad():
        while True:
            actions = policy(obs)

            env.commands[0] = controller.get_commands()
            obs, _, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()
