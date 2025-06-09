import struct
import threading
import signal
import sys
from serial import Serial
import numpy as np

BAUD_RATE = 2000000
PORT = "/dev/ttyAMA2"


class TiComm:
    def __init__(self):
        self.running = True
        self.ser = Serial(PORT, BAUD_RATE)
        self.rx_len = 13  # num of floats
        self.tx_len = 4

        # receive buffer
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.base_lin_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.dof_vel = np.zeros(4, dtype=np.float32)

        # transmit buffer
        self.motor_commands = np.zeros(4, dtype=np.float32)

        # start threads
        self.rx_thread = threading.Thread(target=self._rx)
        self.tx_thread = threading.Thread(target=self._tx)
        self.rx_thread.daemon = True
        self.tx_thread.daemon = True
        self.rx_thread.start()
        self.tx_thread.start()

    def _rx(self):
        while self.running:
            try:
                byte = self.ser.read(2)
                if byte == b"**":
                    packet = self.ser.read(self.rx_len * 4)
                    floats = struct.unpack("<" + "f" * self.rx_len, packet)

                    self.base_ang_vel = np.array(floats[0:3], dtype=np.float32)
                    self.base_lin_vel = np.array(floats[3:6], dtype=np.float32)
                    self.projected_gravity = np.array(floats[6:9], dtype=np.float32)
                    self.dof_vel = np.array(floats[9:13], dtype=np.float32)

                    # change direction
                    self.projected_gravity = -self.projected_gravity
                    self.projected_gravity[-1] -= 1.0

                    self.dof_vel = - self.dof_vel

            except Exception as e:
                if self.running:
                    print(f"RX Error: {e}")
                    break

    def _tx(self):
        while self.running:
            try:
                packet = b"**" + struct.pack("<" + "f" * self.tx_len, *self.motor_commands)
                self.ser.write(packet)
            except Exception as e:
                if self.running:
                    print(f"TX Error: {e}")
                    break

    def stop(self):
        self.running = False
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()

        if hasattr(self, 'rx_thread') and self.rx_thread.is_alive():
            self.rx_thread.join(timeout=1.0)
        if hasattr(self, 'tx_thread') and self.tx_thread.is_alive():
            self.tx_thread.join(timeout=1.0)

    def get_feedback(self):
        return (
            self.base_ang_vel,
            self.base_lin_vel,
            self.projected_gravity,
            self.dof_vel
        )

    def set_motor_commands(self, commands):
        self.motor_commands = commands


if __name__ == "__main__":

    def signal_handler(sig, frame):
        if 'ti' in globals():
            ti.stop()
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_handler)

    ti = TiComm()

    import time

    try:
        while True:
            ang, lin, grav, dof = ti.get_feedback()
            ang_formatted = [f"{x:.3f}" for x in ang]
            lin_formatted = [f"{x:.3f}" for x in lin]
            grav_formatted = [f"{x:.3f}" for x in grav]
            dof_formatted = [f"{x:.3f}" for x in dof]
            print(f"Base Angular Velocity: {ang_formatted}")
            print(f"Base Linear Velocity: {lin_formatted}")
            print(f"Projected Gravity: {grav_formatted}")
            print(f"DOF Velocities: {dof_formatted}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
        ti.stop()
    finally:
        if 'ti' in locals():
            ti.stop()
