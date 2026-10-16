# T12 Remote Kit

A small, hardware-facing Python package extracted from the private
`VITAL_Lab-Omni` recovery source. It reads a T12 transmitter through an R12
receiver and a serial/SBUS-to-USB adapter, normalizes four axes and eight
controls, detects channel mappings, and fails safe when frames stop arriving.

This package is intentionally independent of Omni policy inference, robot motor
control, model files, and vendor SDK code.

## Supported serial formats

- `vendor_decoded`: 35-byte frame, `0x0f` header, sixteen big-endian channel
  words plus one status word.
- `raw_sbus_25`: standard packed 25-byte, 16-channel SBUS frame.

Do not select a format from the transmitter name alone. Confirm the adapter's
actual byte stream.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The runtime dependencies are NumPy, PyYAML, and pyserial.

## Configure safely

1. Identify the adapter with `ls -l /dev/serial/by-id/`.
2. Copy `config/t12_serial.example.yaml` to
   `config/t12_serial.local.yaml`.
3. Fill in the serial path and frame format.
4. Keep the robot disabled or physically supported while measuring channels.
5. Run the read-only detector and follow its prompts:

```bash
t12-detect --config config/t12_serial.local.yaml --switches-only
```

Add `--include-axes` when you are ready to measure stick endpoints. The
detector writes a report; transfer the suggested calibration into the local
YAML and review every channel before connecting the output to a controller.

## Monitor

```bash
t12-monitor --config config/t12_serial.local.yaml
```

The monitor prints normalized axes, switch states, raw channels, link age, and
frame rate. Stop it with Ctrl+C before another process opens the serial port.

## Python API

```python
from t12_remote import T12SerialProvider

with T12SerialProvider("config/t12_serial.local.yaml") as remote:
    snapshot = remote.latest()
    if snapshot is None or not snapshot.connected or snapshot.failsafe:
        command = (0.0, 0.0, 0.0)
    else:
        # Application-owned mapping; this kit does not send robot commands.
        command = (snapshot.y1, -snapshot.x1, -snapshot.x2)
```

`T12RemoteSnapshot` exposes `x1/y1/x2/y2` in `[-1, 1]`,
`button_a..button_d` as `-1/1`, `button_e..button_h` as `-1/0/1`,
raw channels, timestamp, connection state, frame-loss state, and failsafe state.

## Safety boundary

- Treat a missing, stale, malformed, or unplugged serial stream as zero command.
- Never infer channel order from handset labels; calibrate the exact hardware set.
- Do not run mapping detection while another process owns the serial port.
- Keep policy selection and actuator enable logic outside this package.
- The example udev rule contains placeholders and no receiver-specific serial.
- No passwords, control tokens, robot IPs, policy files, or model binaries are
  included.

## Test

```bash
python -m unittest discover -s tests -v
```

Tests exercise both frame formats, fragmented input, calibration, normalization,
stale-frame failsafe behavior, and channel-map detection.

## Distribution status

The repository is published with the owner's authorization. No open-source
software license has been selected yet. Public visibility allows inspection,
but does not by itself grant permission to copy, modify, or redistribute the
code. Authorized coworkers should follow the owner's direct sharing terms; see
`NOTICE.md`.

