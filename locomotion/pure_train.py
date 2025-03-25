import argparse
import os
import sys
import pickle
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pure_env import PureEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [128, 128, 128],
            "critic_hidden_dims": [128, 128, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 50,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "friction": 0.8,
        "num_actions": 4,
        "num_dofs": 20,
        "kp": 5.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 60,  # degree
        "termination_if_pitch_greater_than": 60,
        # base
        "base_init_pos": [0.0, 0.0, 0.12],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "ball_radius": 0.12,
        "episode_length_s": 10.0,
        "resampling_time_s": 4.0,
        "action_scale": 5.0,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        # visualization
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
    }

    obs_cfg = {
        "num_obs": 17,
        "obs_scales": {
            "gravity": 8.0,
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_vel": 0.15,
        }
    }

    reward_cfg = {
        "reward_scales": {
            "vertical": -3.0,
            "vertical_2": -100.0,
            "height": 10.0,
            "track_vel": 1.0,
            "track_ang_vel": 1.0,
            "torque": -1e-5,
            "action_change": -1e-7,
            # "early_termination": -1000.0,
        }
    }

    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-0.001, 0.001],
        "lin_vel_y_range": [-0.001, 0.001],
        "ang_vel_range": [-0.001, 0.001],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="pure")
    parser.add_argument("-B", "--num_envs", type=int, default=8192)
    parser.add_argument("--max_iterations", type=int, default=2500)
    parser.add_argument("-r", "--resume", type=str, default=None)
    args = parser.parse_args()

    gs.init(logging_level="error")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = PureEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    if args.resume is not None:
        try:
            runner.load(f"{args.resume}")
            print(f"Resuming training from {args.resume}")
        except FileNotFoundError:
            print(f"Could not find model checkpoint {args.resume}. Training from scratch.")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
