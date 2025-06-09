import comm
import threading
import signal
import sys
import time

from agent import Agent


class Main:
    def __init__(self):
        self.ti = comm.TiComm()
        model_path = "path/to/model.onnx"
        self.agent = Agent(model_path)
        self.running = True
        signal.signal(signal.SIGINT, self._stop)
        self._loop_thread = threading.Thread(target=self._loop)
        self._loop_thread.daemon = True

    def start(self):
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
            self.ti.set_motor_commands(action)

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
