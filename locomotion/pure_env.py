import torch
import math
import genesis as gs
from genesis.utils.geom import xyz_to_quat, quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import sys
import os


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class PureEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)

        self.zeros = torch.zeros((num_envs, 1), device=self.device, dtype=gs.tc_float)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.01  # run in 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=10),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plane
        self.scene.add_entity(gs.morphs.Plane())

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="assets/pure/urdf/pure.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy()
            ),
        )
        self.robot.set_friction(self.env_cfg["friction"])
        # set_mass_shift(self, mass_shift, link_indices, envs_idx=None):

        # add ball
        self.ball_radius = self.env_cfg["ball_radius"]  # 0.12
        self.ball_init_pos = torch.tensor([0.0, 0.0, self.ball_radius], device=self.device)
        self.ball_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.ball = self.scene.add_entity(
            gs.morphs.Sphere(
                radius=self.ball_radius,
                pos=self.ball_init_pos.cpu().numpy(),
                quat=self.ball_init_quat.cpu().numpy(),
            ),
        )
        # self.ball.get_link(id=0).set_mass(1.0)
        self.ball.set_friction(self.env_cfg["friction"])

        # # add turbulence
        # self.turbulence = self.scene.add_force_field(
        #     gs.force_fields.Turbulence(
        #         strength=10.0,
        #         frequency=0.1,
        #         flow=0.1,
        #     )
        # )
        # self.turbulence.activate()

        # build scene
        self.scene.build(n_envs=num_envs)

        # set gains
        act_joints = ["omni_joint_0", "omni_joint_1", "omni_joint_2", "omni_joint_3"]
        self.act_dof_idx = [self.robot.get_joint(name).dof_idx_local for name in act_joints]  # actuated joints
        self.num_act_dofs = len(self.act_dof_idx)

        self.target_dof_vel = torch.zeros((self.num_envs, self.num_act_dofs), device=self.device, dtype=gs.tc_float)

        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.act_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.act_dof_idx)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.crash_condition = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.early_reset_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)  # rad/s
        self.dof_tor = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.base_euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        self.extras = dict()  # extra information for logging

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        self.target_dof_vel = exec_actions * self.env_cfg["action_scale"]
        self.robot.control_dofs_velocity(self.target_dof_vel, self.act_dof_idx)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.act_dof_idx)
        self.dof_tor[:] = self.robot.get_dofs_force(self.act_dof_idx)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.early_reset_buf = torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.early_reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= self.early_reset_buf

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.base_lin_vel * self.obs_scales["lin_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                self.dof_vel * self.obs_scales["dof_vel"],  # 4
                self.actions,  # 4
            ],
            dim=1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_velocity(self.dof_vel[envs_idx], self.act_dof_idx, envs_idx)
        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)
        # reset ball
        self.ball.set_pos(self.ball_init_pos.unsqueeze(dim=0).repeat(len(envs_idx), 1),
                          zero_velocity=True, envs_idx=envs_idx)
        self.ball.set_quat(self.ball_init_quat.unsqueeze(dim=0).repeat(len(envs_idx), 1),
                           zero_velocity=True, envs_idx=envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.early_reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                    torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_vertical(self):
        return torch.norm(self.projected_gravity[:, :2], dim=1)

    def _reward_vertical_2(self):
        return torch.clip(torch.norm(self.projected_gravity[:, :2], dim=1) - 0.1, 0.0, None)

    def _reward_height(self):
        return self.base_pos[:, 2] - self.ball_radius

    def _reward_track_vel(self):
        error = torch.sum(torch.square(self.base_lin_vel[:, :2] - self.commands[:, :2]), dim=1)
        return torch.exp(-error / 0.1)

    def _reward_track_ang_vel(self):
        error = torch.square(self.base_ang_vel[:, 2] - self.commands[:, 2])  # degree/s
        return torch.exp(-error / 20.0)

    def _reward_torque(self):
        return torch.sum(torch.square(self.dof_tor), dim=1)

    def _reward_action_change(self):
        return torch.sum(torch.square((self.actions - self.last_actions) / self.dt), dim=1)

    def _reward_early_termination(self):
        return self.early_reset_buf.float()

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)