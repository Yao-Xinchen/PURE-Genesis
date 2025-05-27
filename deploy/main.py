import onnxruntime as ort
import comm
import numpy as np
import threading
import asyncio
import signal
import sys
import time


class Agent():
    def __init__(self, model_path):
        self._session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_names = [input.name for input in self._session.get_inputs()]
        print('Actor input names:', input_names)
        output_names = [output.name for output in self._session.get_outputs()]
        print('Actor output names:', output_names)
        self._input_name = input_names[0]
        self._output_name = output_names[0]

        self._observation = np.zeros((1, 16), dtype=np.float32)
        self._action = np.zeros((1, 4), dtype=np.float32)

        self._dt = 0.02
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
                observation = self._observation.copy()
            input = {self._input_name: np.concatenate((observation, self._action), axis=1)}
            output = self._session.run([self._output_name], input)[0]
            with self._lock:
                self._action = output
            await asyncio.sleep(self._dt)

    def set_ang_vel(self, ang_vel):
        with self._lock:
            self._observation[0, 0:3] = ang_vel * 0.25

    def set_lin_vel(self, lin_vel):
        with self._lock:
            self._observation[0, 3:6] = lin_vel * 2.0

    def set_gravity(self, gravity):
        with self._lock:
            self._observation[0, 6:9] = gravity

    def set_commands(self, commands):
        with self._lock:
            self._observation[0, 9:12] = commands * 0.15

    def set_dof_vel(self, dof_vel):
        with self._lock:
            self._observation[0, 12:16] = dof_vel * 0.15 * 180.0 / np.pi

    def get_action(self):
        with self._lock:
            return self._action[0].copy() * 100.0 * np.pi / 180.0

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


class Main():
    def __init__(self):
        self.ti = comm.TiComm()
        model_path = "path/to/model.onnx"
        self.agent = Agent(model_path)
        self.running = True
        signal.signal(signal.SIGINT, self._stop)

    def start(self):
        self._loop_thread = threading.Thread(target=self._loop)
        self._loop_thread.daemon = True
        self._loop_thread.start()

        try:
            while self.running:
                time.sleep(0.5)
        except Exception as e:
            print(f"Error in main thread: {e}")
        finally:
            self._cleanup()

    def _loop(self):
        while self.running:
            ang_vel, lin_vel, gravity, dof_vel = self.ti.get_feedback()
            self.agent.set_ang_vel(ang_vel)
            self.agent.set_lin_vel(lin_vel)
            self.agent.set_gravity(gravity)
            self.agent.set_dof_vel(dof_vel)

            action = self.agent.get_action()
            self.ti.set_commands(action)

    def _stop(self, signum, frame):
        print("\nShutting down...")
        self.running = False

    def _cleanup(self):
        if hasattr(self, '_loop_thread') and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=2.0)

        if hasattr(self, 'agent'):
            self.agent.shutdown()

        print("Clean shutdown complete")
        sys.exit(0)


if __name__ == "__main__":
    m = Main()
    m.start()
