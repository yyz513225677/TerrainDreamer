"""
IMU Driver — Pluggable Interface
==================================
Abstract base + placeholder implementation.
Leonardo: replace `PlaceholderIMU` with your specific model once decided.

Supported patterns:
  - Serial (UART): Most common for Xsens, Microstrain, Wit Motion
  - SPI: Some embedded IMUs (Bosch, InvenSense)
  - USB/HID: Some dev kits
  - ROS topic: If running alongside ROS
"""

import time
import threading
import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class IMUReading:
    """Single IMU measurement."""
    timestamp: float             # Unix time
    # Accelerometer (m/s²) in body frame
    accel: np.ndarray            # [ax, ay, az]
    # Gyroscope (rad/s) in body frame
    gyro: np.ndarray             # [wx, wy, wz]
    # Magnetometer (μT) — optional
    mag: Optional[np.ndarray] = None   # [mx, my, mz]
    # If the IMU provides onboard orientation estimate (quaternion)
    orientation_quat: Optional[np.ndarray] = None  # [w, x, y, z]
    # Temperature (°C) — useful for bias compensation
    temperature: Optional[float] = None


class IMUDriverBase(ABC):
    """Abstract IMU interface that any hardware driver must implement."""

    @abstractmethod
    def start(self):
        """Begin data acquisition."""
        ...

    @abstractmethod
    def stop(self):
        """Stop data acquisition and release resources."""
        ...

    @abstractmethod
    def get_reading(self, timeout: float = 0.1) -> Optional[IMUReading]:
        """Get the latest IMU reading (blocking with timeout)."""
        ...

    @abstractmethod
    def get_frequency(self) -> float:
        """Return actual data rate in Hz."""
        ...


class PlaceholderIMU(IMUDriverBase):
    """
    Generates synthetic IMU data for development/testing.
    Replace this with your real hardware driver.

    Simulates:
      - Gravity on Z-axis
      - Small random noise on accel/gyro
      - Configurable bias
    """

    def __init__(self, frequency_hz: float = 200.0):
        self.frequency_hz = frequency_hz
        self._dt = 1.0 / frequency_hz
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._queue: deque[IMUReading] = deque(maxlen=500)

        # Simulated biases (typical MEMS values)
        self._accel_bias = np.array([0.02, -0.01, 0.03])   # m/s²
        self._gyro_bias = np.array([0.001, -0.0005, 0.0008])  # rad/s
        self._accel_noise_std = 0.05   # m/s²
        self._gyro_noise_std = 0.005   # rad/s

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._generate_loop, daemon=True, name="imu-placeholder"
        )
        self._thread.start()
        print(f"[IMU-Placeholder] Running at {self.frequency_hz} Hz")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("[IMU-Placeholder] Stopped")

    def get_reading(self, timeout: float = 0.1) -> Optional[IMUReading]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._queue:
                return self._queue.popleft()
            time.sleep(0.001)
        return None

    def get_frequency(self) -> float:
        return self.frequency_hz

    def _generate_loop(self):
        while self._running:
            reading = IMUReading(
                timestamp=time.time(),
                accel=np.array([0.0, 0.0, 9.81]) + self._accel_bias
                    + np.random.randn(3) * self._accel_noise_std,
                gyro=self._gyro_bias
                    + np.random.randn(3) * self._gyro_noise_std,
                mag=np.array([25.0, 5.0, -45.0])
                    + np.random.randn(3) * 0.5,
                temperature=25.0 + np.random.randn() * 0.1,
            )
            self._queue.append(reading)
            time.sleep(self._dt)


class SerialIMU(IMUDriverBase):
    """
    Template for serial-port IMU (UART/USB-Serial).

    Common IMU models and their typical protocols:
      - Xsens MTi:     MT binary protocol
      - Microstrain 3DM: MIP protocol
      - Wit Motion:     Wit protocol (0x55 header)
      - VectorNav VN-100: ASCII or binary
      - Bosch BNO055:   UART/I2C register protocol

    Leonardo: Implement `_parse_packet()` for your specific IMU model.
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        frequency_hz: float = 200.0,
    ):
        self.port = port
        self.baudrate = baudrate
        self.frequency_hz = frequency_hz
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._queue: deque[IMUReading] = deque(maxlen=500)
        self._serial = None

    def start(self):
        try:
            import serial
        except ImportError:
            raise ImportError("pip install pyserial")

        self._serial = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=0.01,
        )
        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name="imu-serial"
        )
        self._thread.start()
        print(f"[IMU-Serial] Opened {self.port} @ {self.baudrate}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._serial and self._serial.is_open:
            self._serial.close()

    def get_reading(self, timeout: float = 0.1) -> Optional[IMUReading]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._queue:
                return self._queue.popleft()
            time.sleep(0.001)
        return None

    def get_frequency(self) -> float:
        return self.frequency_hz

    def _read_loop(self):
        """Read serial data and parse into IMUReading objects."""
        buffer = bytearray()
        while self._running:
            if self._serial.in_waiting > 0:
                buffer.extend(self._serial.read(self._serial.in_waiting))
                # Try to extract complete packets from buffer
                reading, consumed = self._parse_packet(buffer)
                if reading is not None:
                    self._queue.append(reading)
                    buffer = buffer[consumed:]
            else:
                time.sleep(0.001)

    def _parse_packet(self, buffer: bytearray):
        """
        Parse IMU-specific packet from serial buffer.

        TODO: Implement for your specific IMU model.

        Returns:
            (IMUReading or None, bytes consumed)
        """
        # ============================================================
        # EXAMPLE: Wit Motion protocol (0x55 header)
        # Packet: [0x55] [Type] [Data×8] [Checksum] = 11 bytes
        #
        # Type 0x51 = Acceleration
        # Type 0x52 = Angular velocity
        # Type 0x53 = Angle (Euler)
        # ============================================================
        raise NotImplementedError(
            "Implement _parse_packet() for your specific IMU model. "
            "See docstring for protocol examples."
        )
