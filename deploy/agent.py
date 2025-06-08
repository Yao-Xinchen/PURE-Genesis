import asyncio
import threading

import numpy as np
import onnxruntime as ort


class Agent:
    def __init__(self, model_path):
        self._obs_scales = {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_vel": 0.1,
            "gravity": 10.0,
        }

        self._action_scale = 10.0

        # self.num_obs = 20
        self.num_obs = 13
        self.num_actions = 4

        self._session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_names = [input.name for input in self._session.get_inputs()]
        print('Actor input names:', input_names)
        output_names = [output.name for output in self._session.get_outputs()]
        print('Actor output names:', output_names)
        self._input_name = input_names[0]
        self._output_name = output_names[0]

        self._observation = np.zeros((1, self.num_obs), dtype=np.float32)
        self._action = np.zeros((1, self.num_actions), dtype=np.float32)

        self._ang_vel = np.zeros((1, 3), dtype=np.float32)
        self._lin_vel = np.zeros((1, 3), dtype=np.float32)
        self._grav = np.zeros((1, 3), dtype=np.float32)
        self._dof_vel = np.zeros((1, 4), dtype=np.float32)
        self._commands = np.zeros((1, 3), dtype=np.float32)

        self._dt = 0.01
        self._lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_async_loop)
        self._thread.daemon = True
        self._thread.start()

    def _start_async_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._run())
        self._loop.close()

    async def _run(self):
        while True:
            with self._lock:
                self._observation = np.concatenate(
                    (
                        self._ang_vel,
                        # self._lin_vel,
                        self._grav[:, :2],
                        # self._commands,
                        self._dof_vel,
                        self._action,
                    ), axis=1
                )

            output = self._session.run(
                [self._output_name],
                {self._input_name: self._observation}
            )[0]

            with self._lock:
                self._action = output

            await asyncio.sleep(self._dt)

    def set_ang_vel(self, ang_vel):
        with self._lock:
            self._ang_vel[0] = ang_vel * self._obs_scales["ang_vel"]

    def set_lin_vel(self, lin_vel):
        with self._lock:
            self._lin_vel[0] = lin_vel * self._obs_scales["lin_vel"]

    def set_gravity(self, gravity):
        with self._lock:
            self._grav[0] = gravity * self._obs_scales["gravity"]

    def set_commands(self, commands):
        with self._lock:
            self._commands[0] = commands

    def set_dof_vel(self, dof_vel):
        with self._lock:
            self._dof_vel[0] = dof_vel * self._obs_scales["dof_vel"]

    def get_action(self):
        with self._lock:
            return self._action[0].copy() * self._action_scale

    def shutdown(self):
        if hasattr(self, '_loop') and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)

        if hasattr(self, '_thread') and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    async def _shutdown(self):
        tasks = [t for t in asyncio.all_tasks(self._loop) if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._loop.stop()
