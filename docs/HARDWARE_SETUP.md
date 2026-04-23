# Hardware Setup Guide

## 1. Velodyne VLP-32C — Full Connection Guide

### Physical Connection

```
[VLP-32C] ──── Interface Box ──── Ethernet ──── [Your PC]
                   │
                   └── 12V Power Supply (16W typical)
```

The VLP-32C ships with an **Interface Box** that has:
- Power input (12V DC)
- Ethernet port (RJ45)
- GPS PPS input (optional)

### Network Configuration

The VLP-32C has a **factory default IP: 192.168.1.201**.
Your computer's NIC must be on the same subnet.

```bash
# Step 1: Identify your Ethernet interface
ip link show
# Look for your wired adapter, e.g., eth0, enp3s0, eno1

# Step 2: Add an IP on the VLP-32C's subnet
sudo ip addr add 192.168.1.100/24 dev eth0
# (Replace 'eth0' with your interface name)

# Step 3: Verify connectivity
ping 192.168.1.201
# You should see replies if the VLP-32C is powered on

# Step 4: Access the web interface (optional, for configuration)
# Open a browser to: http://192.168.1.201
# Here you can set RPM, return mode, FOV, etc.
```

### Firewall Rules

```bash
# Allow incoming UDP on port 2368 (data) and 8308 (position)
sudo ufw allow 2368/udp
sudo ufw allow 8308/udp

# Or if using iptables:
sudo iptables -A INPUT -p udp --dport 2368 -j ACCEPT
sudo iptables -A INPUT -p udp --dport 8308 -j ACCEPT
```

### Verify Data is Arriving

```bash
# Check for UDP packets (should see continuous output)
sudo tcpdump -i eth0 port 2368 -c 10

# Expected output:
# 192.168.1.201.2368 > 255.255.255.255.2368: UDP, length 1206
```

### How the Raw Data Works

The VLP-32C sends **1206-byte UDP packets** at ~754 packets/second (at 600 RPM):

```
┌──────────────────────────────────────────────────────────┐
│                    1206-byte Packet                       │
├───────────┬───────────┬─────────────────┬────────────────┤
│  Block 0  │  Block 1  │  ... Block 11   │ Timestamp(4B)  │
│  100 B    │  100 B    │                 │ Factory(2B)    │
├───────────┴───────────┴─────────────────┴────────────────┤
│                                                          │
│  Each Block (100 bytes):                                 │
│  ┌────────┬──────────┬───────────────────────────────┐   │
│  │ Flag   │ Azimuth  │  32 Channels × 3 bytes        │   │
│  │ 0xFFEE │ 2 bytes  │  [distance(2B)][reflectivity] │   │
│  │ 2 bytes│ 0.01°    │  distance in 2mm units        │   │
│  └────────┴──────────┴───────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

Our driver (`sensors/velodyne_vlp32.py`) reads these raw UDP packets,
parses all 12 blocks × 32 channels per packet, converts spherical
coordinates to Cartesian (x, y, z), and accumulates points until a
full 360° revolution is detected (azimuth wraps around).

### Test the Connection

```bash
cd terrain_dreamer/
python scripts/test_velodyne.py
```

Expected output:
```
Velodyne VLP-32C Connection Test
Listening for data on UDP port 2368...

  Scan: 57344 points | X range: [-45.2, 52.3] | Y range: [-48.1, 49.7] | Z range: [-2.1, 12.5]
  Scan: 57120 points | ...

[30s] Stats: {'total_packets': 22620, 'total_scans': 30, 'dropped_packets': 0}
SUCCESS: VLP-32C is working!
```

### Persistent Network Setup

To avoid running `ip addr add` every boot:

```bash
# Ubuntu 22.04+ with Netplan
sudo nano /etc/netplan/01-velodyne.yaml
```

```yaml
network:
  version: 2
  ethernets:
    eth0:                      # Your interface name
      dhcp4: false
      addresses:
        - 192.168.1.100/24
      # No gateway — this is a dedicated sensor network
```

```bash
sudo netplan apply
```

---

## 2. IMU Setup (Placeholder)

Leonardo: update this section once you have your IMU model.

### Common IMU Models for Off-Road AV

| Model              | Protocol      | Interface   | Price Range |
|--------------------|---------------|-------------|-------------|
| Xsens MTi-630      | MT binary     | USB/UART    | $$$$        |
| Microstrain 3DM-GQ7| MIP          | USB/SPI     | $$$$        |
| VectorNav VN-100   | ASCII/Binary  | UART/SPI    | $$$         |
| Wit Motion BWT901CL| Wit 0x55      | UART/BT     | $           |
| Bosch BNO055       | I2C/UART      | I2C/UART    | $           |
| InvenSense ICM-42688| SPI/I2C      | SPI         | $           |

### What You Need to Implement

In `sensors/imu_driver.py`, implement the `SerialIMU._parse_packet()` method:

1. Read raw bytes from serial port
2. Find packet header (model-specific)
3. Parse accelerometer data (m/s²)
4. Parse gyroscope data (rad/s)
5. Optionally parse magnetometer, temperature, orientation
6. Package into `IMUReading` dataclass

### Calibration

Once the IMU is connected, you'll need the **LiDAR-to-IMU extrinsic transform**
(`T_imu_lidar` in config.yaml). Measure or calibrate:
- Translation: physical offset between IMU and LiDAR center (meters)
- Rotation: alignment between IMU axes and LiDAR axes

---

## 3. Vehicle Control Interface

### CAN Bus (Recommended)

```bash
# Setup SocketCAN
sudo ip link set can0 type can bitrate 500000
sudo ip link set up can0

# Verify
candump can0
```

### Serial (Simpler vehicles / RC cars)

```bash
# Check port
ls /dev/ttyUSB* /dev/ttyACM*

# Test
screen /dev/ttyUSB0 115200
```

---

## 4. Full System Wiring

```
                    ┌─────────────┐
                    │    PC       │
                    │  (GPU)     │
                    └──┬───┬──┬──┘
                       │   │  │
              Ethernet │   │  │ USB/Serial
                       │   │  │
                ┌──────┘   │  └──────┐
                │          │         │
        ┌───────▼──┐  ┌───▼───┐  ┌──▼──────┐
        │ VLP-32C  │  │  CAN  │  │   IMU   │
        │ LiDAR    │  │ Bus   │  │  (TBD)  │
        │          │  │       │  │         │
        └──────────┘  └───┬───┘  └─────────┘
                          │
                   ┌──────▼──────┐
                   │   Vehicle   │
                   │  Steering   │
                   │  Throttle   │
                   │  Brake      │
                   └─────────────┘
```
