"""
Velodyne VLP-32C Raw UDP Driver
================================
Reads raw data packets directly from the VLP-32C LiDAR via UDP socket.
No ROS dependency — pure Python with numpy acceleration.

Hardware Setup
--------------
1. Connect VLP-32C to your computer via Ethernet
2. The VLP-32C ships with factory IP: 192.168.1.201
3. Configure your NIC to be on the same subnet:
       sudo ip addr add 192.168.1.100/24 dev eth0
4. Verify connectivity:
       ping 192.168.1.201
5. Access web interface at http://192.168.1.201 to configure RPM, FOV, return mode

Packet Format (from Velodyne VLP-32C User Manual)
--------------------------------------------------
Data packet (port 2368): 1206 bytes total
  - 12 data blocks × 100 bytes each = 1200 bytes
  - 4 bytes timestamp (microseconds from top of hour)
  - 2 bytes factory info (return mode + product ID)

Each data block (100 bytes):
  - 2 bytes flag: 0xFFEE
  - 2 bytes azimuth (0.01° resolution, 0–35999)
  - 32 channels × 3 bytes each:
      - 2 bytes distance (2mm resolution)
      - 1 byte reflectivity (0–255)
"""

import socket
import struct
import threading
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable

# ---------------------------------------------------------------------------
# Constants from VLP-32C spec
# ---------------------------------------------------------------------------
DATA_PORT = 2368
POSITION_PORT = 8308
PACKET_SIZE = 1206
BLOCK_SIZE = 100
BLOCKS_PER_PACKET = 12
CHANNELS_PER_BLOCK = 32
BLOCK_FLAG = 0xFFEE
DISTANCE_RESOLUTION = 0.002      # 2 mm per unit
AZIMUTH_RESOLUTION = 0.01        # 0.01 degrees per unit
ROTATION_MAX_UNITS = 36000        # 360.00 degrees

# VLP-32C channel vertical angles (elevation) in degrees
# From the official calibration data
VLP32C_ELEVATION_DEG = np.array([
    -25.0, -15.639, -11.310, -8.843,
     -7.254, -6.148, -5.333, -4.667,
     -4.0,  -3.667, -3.333, -3.0,
     -2.667, -2.333, -2.0,  -1.667,
     -1.333, -1.0,  -0.667, -0.333,
      0.0,   0.333,  0.667,  1.0,
      1.333,  1.667,  2.333,  3.333,
      4.667,  7.0,   10.333, 15.0
], dtype=np.float64)

VLP32C_ELEVATION_RAD = np.deg2rad(VLP32C_ELEVATION_DEG)

# Azimuth correction per channel (factory calibration)
VLP32C_AZIMUTH_CORRECTION_DEG = np.array([
     1.4,  -4.2,   1.4,  -1.4,
     1.4,  -1.4,   4.2,  -1.4,
     1.4,  -4.2,   1.4,  -1.4,
     4.2,  -1.4,   4.2,  -1.4,
     1.4,  -4.2,   1.4,  -4.2,
     4.2,  -1.4,   1.4,  -1.4,
     1.4,  -1.4,   1.4,  -4.2,
     4.2,  -1.4,   4.2,  -1.4,
], dtype=np.float64)

VLP32C_AZIMUTH_CORRECTION_RAD = np.deg2rad(VLP32C_AZIMUTH_CORRECTION_DEG)


@dataclass
class PointCloud:
    """Single LiDAR scan (one full 360° revolution)."""
    timestamp: float                        # Unix time
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 4)))
    # Each row: [x, y, z, intensity]
    # Coordinate frame: x=forward, y=left, z=up (right-hand, NED variant)
    ring_ids: Optional[np.ndarray] = None   # Channel index per point
    azimuths: Optional[np.ndarray] = None   # Azimuth per point (deg)

    @property
    def num_points(self) -> int:
        return self.points.shape[0]

    @property
    def xyz(self) -> np.ndarray:
        return self.points[:, :3]

    @property
    def intensity(self) -> np.ndarray:
        return self.points[:, 3]


class VelodyneVLP32C:
    """
    Real-time driver for Velodyne VLP-32C.

    Reads raw UDP packets, assembles full 360° scans, and delivers
    PointCloud objects through a callback or a thread-safe queue.

    Usage
    -----
    >>> def on_scan(pc: PointCloud):
    ...     print(f"Got {pc.num_points} points")
    ...
    >>> driver = VelodyneVLP32C(on_scan_callback=on_scan)
    >>> driver.start()
    >>> # ... runs until ...
    >>> driver.stop()
    """

    def __init__(
        self,
        data_port: int = DATA_PORT,
        listen_ip: str = "0.0.0.0",
        max_range: float = 100.0,
        min_range: float = 0.5,
        on_scan_callback: Optional[Callable[[PointCloud], None]] = None,
        queue_size: int = 10,
    ):
        self.data_port = data_port
        self.listen_ip = listen_ip
        self.max_range = max_range
        self.min_range = min_range
        self.callback = on_scan_callback

        # Thread-safe scan queue (for polling mode)
        self.scan_queue: deque[PointCloud] = deque(maxlen=queue_size)

        # Internal state
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Accumulate points for one full revolution
        self._current_points = []
        self._current_rings = []
        self._current_azimuths = []
        self._last_azimuth = -1.0
        self._scan_start_time = time.time()

        # Pre-compute trig tables for elevation
        self._cos_elev = np.cos(VLP32C_ELEVATION_RAD)
        self._sin_elev = np.sin(VLP32C_ELEVATION_RAD)

        # Stats
        self.total_packets = 0
        self.total_scans = 0
        self.dropped_packets = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Open UDP socket and begin receiving in background thread."""
        if self._running:
            return

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Increase receive buffer to avoid drops at high rotation speeds
        self._sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024  # 4 MB
        )
        self._sock.settimeout(1.0)
        self._sock.bind((self.listen_ip, self.data_port))

        self._running = True
        self._thread = threading.Thread(
            target=self._receive_loop, daemon=True, name="vlp32c-rx"
        )
        self._thread.start()
        print(f"[VLP-32C] Listening on {self.listen_ip}:{self.data_port}")

    def stop(self):
        """Stop receiving and close socket."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._sock is not None:
            self._sock.close()
        print(f"[VLP-32C] Stopped. Packets={self.total_packets}, "
              f"Scans={self.total_scans}, Dropped={self.dropped_packets}")

    def get_scan(self, timeout: float = 1.0) -> Optional[PointCloud]:
        """Poll for the next complete scan (blocking with timeout)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.scan_queue:
                return self.scan_queue.popleft()
            time.sleep(0.005)
        return None

    # ------------------------------------------------------------------
    # Packet Parsing
    # ------------------------------------------------------------------

    def _receive_loop(self):
        """Background thread: read UDP packets and assemble scans."""
        while self._running:
            try:
                data, _ = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) != PACKET_SIZE:
                self.dropped_packets += 1
                continue

            self.total_packets += 1
            self._parse_packet(data)

    def _parse_packet(self, raw: bytes):
        """
        Parse one 1206-byte data packet into XYZ + intensity points.

        Packet layout:
          [Block0 (100B)][Block1]...[Block11] [Timestamp(4B)] [Factory(2B)]

        Block layout:
          [Flag(2B)][Azimuth(2B)][Ch0 Dist(2B) Refl(1B)]...[Ch31 Dist(2B) Refl(1B)]
        """
        # Extract timestamp (microseconds past the hour)
        timestamp_us = struct.unpack_from("<I", raw, 1200)[0]
        # Factory bytes: raw[1204] = return mode, raw[1205] = product ID
        # return_mode = raw[1204]  # 0x37=strongest, 0x38=last, 0x39=dual

        for block_idx in range(BLOCKS_PER_PACKET):
            offset = block_idx * BLOCK_SIZE

            # Verify flag
            flag = struct.unpack_from("<H", raw, offset)[0]
            if flag != BLOCK_FLAG:
                continue

            # Azimuth for this block (in 0.01° units)
            azimuth_raw = struct.unpack_from("<H", raw, offset + 2)[0]
            azimuth_deg = azimuth_raw * AZIMUTH_RESOLUTION

            # Detect scan boundary: azimuth wrapped around 360° → new scan
            if self._last_azimuth > 300.0 and azimuth_deg < 60.0:
                self._emit_scan()

            self._last_azimuth = azimuth_deg

            # Parse 32 channels in this block
            azimuth_rad = np.deg2rad(azimuth_deg)
            ch_offset = offset + 4  # skip flag(2) + azimuth(2)

            for ch in range(CHANNELS_PER_BLOCK):
                # Distance: 2 bytes little-endian, in 2mm units
                dist_raw = struct.unpack_from("<H", raw, ch_offset)[0]
                reflectivity = raw[ch_offset + 2]
                ch_offset += 3

                distance = dist_raw * DISTANCE_RESOLUTION

                # Range gate
                if distance < self.min_range or distance > self.max_range:
                    continue
                if dist_raw == 0:
                    continue

                # Spherical → Cartesian
                # Apply per-channel azimuth correction
                az = azimuth_rad + VLP32C_AZIMUTH_CORRECTION_RAD[ch]
                cos_elev = self._cos_elev[ch]
                sin_elev = self._sin_elev[ch]

                xy_dist = distance * cos_elev
                x = xy_dist * np.sin(az)   # forward
                y = xy_dist * np.cos(az)   # left
                z = distance * sin_elev    # up

                self._current_points.append([x, y, z, reflectivity / 255.0])
                self._current_rings.append(ch)
                self._current_azimuths.append(azimuth_deg)

    def _emit_scan(self):
        """Package accumulated points into a PointCloud and deliver it."""
        if len(self._current_points) < 100:
            # Skip degenerate scans
            self._current_points.clear()
            self._current_rings.clear()
            self._current_azimuths.clear()
            return

        pc = PointCloud(
            timestamp=time.time(),
            points=np.array(self._current_points, dtype=np.float32),
            ring_ids=np.array(self._current_rings, dtype=np.int32),
            azimuths=np.array(self._current_azimuths, dtype=np.float32),
        )

        # Deliver via callback
        if self.callback is not None:
            try:
                self.callback(pc)
            except Exception as e:
                print(f"[VLP-32C] Callback error: {e}")

        # Also push to queue
        self.scan_queue.append(pc)
        self.total_scans += 1

        # Reset accumulator
        self._current_points.clear()
        self._current_rings.clear()
        self._current_azimuths.clear()
        self._scan_start_time = time.time()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "total_packets": self.total_packets,
            "total_scans": self.total_scans,
            "dropped_packets": self.dropped_packets,
            "queue_depth": len(self.scan_queue),
        }


class VelodynePlayback:
    """
    Replay saved point clouds (e.g., from RELLIS-3D .bin files or .npy)
    through the same PointCloud interface.

    Usage
    -----
    >>> playback = VelodynePlayback(data_dir="data/sequences/00/velodyne/")
    >>> for pc in playback:
    ...     process(pc)
    """

    def __init__(self, data_dir: str, format: str = "bin", loop: bool = False):
        import os
        self.data_dir = data_dir
        self.format = format
        self.loop = loop
        self.files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith(f".{format}")
        ])
        self._index = 0
        print(f"[Playback] Found {len(self.files)} frames in {data_dir}")

    def __iter__(self):
        return self

    def __next__(self) -> PointCloud:
        if self._index >= len(self.files):
            if self.loop:
                self._index = 0
            else:
                raise StopIteration

        filepath = self.files[self._index]
        self._index += 1

        if self.format == "bin":
            # KITTI / RELLIS-3D format: N×4 float32 (x, y, z, intensity)
            points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
        elif self.format == "npy":
            points = np.load(filepath).astype(np.float32)
        else:
            raise ValueError(f"Unknown format: {self.format}")

        return PointCloud(
            timestamp=time.time(),
            points=points,
        )

    def reset(self):
        self._index = 0
