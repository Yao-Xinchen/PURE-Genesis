import struct
import threading
import signal
from serial import Serial

BAUD_RATE = 115200
PORT = "/dev/ttyAMA1"


class TiComm:
    def __init__(self):
        self.running = True
        self.ser = Serial(PORT, BAUD_RATE)
        self.rx_len = 13  # num of floats
        self.tx_len = 4

        # receive buffer
        self.base_ang_vel = [0.0, 0.0, 0.0]
        self.base_lin_vel = [0.0, 0.0, 0.0]
        self.projected_gravity = [0.0, 0.0, 9.81]
        self.dof_vel = [0.0, 0.0, 0.0, 0.0]

        # transmit buffer
        self.commands = [0.0, 0.0, 0.0, 0.0]

        # start threads
        self.rx_thread = threading.Thread(target=self._rx)
        self.tx_thread = threading.Thread(target=self._tx)
        self.rx_thread.daemon = True
        self.tx_thread.daemon = True
        self.rx_thread.start()
        self.tx_thread.start()
        signal.signal(signal.SIGINT, self._stop)

    def _rx(self):
        while self.running:
            try:
                byte = self.ser.read(2)
                if byte == b"**":
                    packet = self.ser.read(self.rx_len * 4)
                    floats = struct.unpack("<" + "f" * self.rx_len, packet)

                    self.base_ang_vel = floats[0:3]
                    self.base_lin_vel = floats[3:6]
                    self.projected_gravity = floats[6:9]
                    self.dof_vel = floats[9:13]
            except Exception as e:
                if self.running:
                    print(f"RX Error: {e}")
                    break

    def _tx(self):
        while self.running:
            try:
                packet = b"**" + struct.pack("<" + "f" * self.tx_len, *self.commands)
                self.ser.write(packet)
            except Exception as e:
                if self.running:
                    print(f"TX Error: {e}")
                    break

    def _stop(self, signum, frame):
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

    def set_commands(self, commands):
        self.commands = commands


if __name__ == "__main__":
    ti = TiComm()

    import time

    while True:
        ang, _, _, _ = ti.get_feedback()
        print(f"Base Angular Velocity: {ang}")

        time.sleep(0.5)
