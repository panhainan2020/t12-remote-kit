"""Non-blocking T12/R12 serial input for robotics and simulation applications.

Two wire formats are deliberately kept separate:

* ``vendor35`` is the 35-byte, 115200-baud frame emitted by a decoded-channel
  SBUS-to-USB adapter.  It contains a 0x0f byte followed by
  seventeen big-endian 16-bit words (sixteen channels and one status word).
* ``sbus25`` is an ordinary 25-byte packed SBUS frame.

Channel meanings and calibration live in YAML.  The decoders never hard-code
T12 switch/channel assignments.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import threading
import time
from typing import Any, Iterable

import numpy as np
import yaml


@dataclass(frozen=True)
class DecodedT12Frame:
    channels: tuple[int, ...]
    frame_lost: bool = False
    failsafe: bool = False
    status: int = 0


@dataclass(frozen=True)
class T12RemoteSnapshot:
    x1: float
    y1: float
    x2: float
    y2: float
    button_a: int
    button_b: int
    button_c: int
    button_d: int
    button_e: int
    button_f: int
    button_g: int
    button_h: int
    timestamp: float
    connected: bool
    failsafe: bool
    frame_lost: bool = False
    raw_channels: tuple[int, ...] = ()


class VendorDecoded35Decoder:
    """Decode the lab converter's fixed 35-byte channel frame."""

    FRAME_SIZE = 35
    HEADER = 0x0F

    def __init__(
        self,
        *,
        lost_mask: int = 0,
        failsafe_mask: int = 0,
        max_channel_value: int = 2047,
    ) -> None:
        self.lost_mask = int(lost_mask)
        self.failsafe_mask = int(failsafe_mask)
        self.max_channel_value = int(max_channel_value)
        self._buffer = bytearray()

    def decode(self, frame: bytes) -> DecodedT12Frame:
        if len(frame) != self.FRAME_SIZE or frame[0] != self.HEADER:
            raise ValueError("vendor35 frame must be 35 bytes with 0x0f header")
        words = tuple(
            int.from_bytes(frame[index : index + 2], "big")
            for index in range(1, self.FRAME_SIZE, 2)
        )
        channels, status = words[:16], words[16]
        if any(value < 0 or value > self.max_channel_value for value in channels):
            raise ValueError("vendor35 channel value is outside configured range")
        return DecodedT12Frame(
            channels=channels,
            frame_lost=bool(self.lost_mask and status & self.lost_mask),
            failsafe=bool(self.failsafe_mask and status & self.failsafe_mask),
            status=status,
        )

    def feed(self, data: bytes) -> list[DecodedT12Frame]:
        self._buffer.extend(data)
        output: list[DecodedT12Frame] = []
        while True:
            try:
                header = self._buffer.index(self.HEADER)
            except ValueError:
                self._buffer.clear()
                break
            if header:
                del self._buffer[:header]
            if len(self._buffer) < self.FRAME_SIZE:
                break
            candidate = bytes(self._buffer[: self.FRAME_SIZE])
            try:
                decoded = self.decode(candidate)
            except ValueError:
                del self._buffer[0]
                continue
            output.append(decoded)
            del self._buffer[: self.FRAME_SIZE]
        return output


class RawSbus25Decoder:
    """Decode standard 16-channel, 11-bit packed SBUS frames."""

    FRAME_SIZE = 25
    HEADER = 0x0F

    def __init__(self) -> None:
        self._buffer = bytearray()

    def decode(self, frame: bytes) -> DecodedT12Frame:
        if len(frame) != self.FRAME_SIZE or frame[0] != self.HEADER:
            raise ValueError("SBUS frame must be 25 bytes with 0x0f header")
        packed = int.from_bytes(frame[1:23], "little")
        channels = tuple((packed >> (11 * index)) & 0x07FF for index in range(16))
        flags = frame[23]
        return DecodedT12Frame(
            channels=channels,
            frame_lost=bool(flags & 0x04),
            failsafe=bool(flags & 0x08),
            status=flags,
        )

    def feed(self, data: bytes) -> list[DecodedT12Frame]:
        self._buffer.extend(data)
        output: list[DecodedT12Frame] = []
        while True:
            try:
                header = self._buffer.index(self.HEADER)
            except ValueError:
                self._buffer.clear()
                break
            if header:
                del self._buffer[:header]
            if len(self._buffer) < self.FRAME_SIZE:
                break
            candidate = bytes(self._buffer[: self.FRAME_SIZE])
            output.append(self.decode(candidate))
            del self._buffer[: self.FRAME_SIZE]
        return output


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


class T12ChannelMapper:
    """Apply YAML channel calibration and emit the SDK-compatible payload."""

    AXES = ("x1", "y1", "x2", "y2")
    BUTTONS = tuple(f"button_{name}" for name in "abcdefgh")

    def __init__(self, channels: dict[str, Any]) -> None:
        self.channels = channels
        missing = set(self.AXES + self.BUTTONS) - set(channels)
        if missing:
            raise ValueError(f"T12 channel map is missing: {sorted(missing)}")

    @staticmethod
    def _raw(frame: DecodedT12Frame, spec: dict[str, Any]) -> float:
        index = int(spec["index"])
        if index < 0 or index >= len(frame.channels):
            raise ValueError(f"T12 channel index {index} is out of range")
        return float(frame.channels[index])

    def _axis(self, frame: DecodedT12Frame, name: str) -> float:
        spec = _mapping(self.channels[name], name=name)
        raw = self._raw(frame, spec)
        minimum = float(spec["min"])
        center = float(spec["center"])
        maximum = float(spec["max"])
        if not minimum < center < maximum:
            raise ValueError(f"{name} calibration must satisfy min < center < max")
        value = (
            (raw - center) / (maximum - center)
            if raw >= center
            else (raw - center) / (center - minimum)
        )
        if bool(spec.get("reversed", False)):
            value = -value
        return float(np.clip(value, -1.0, 1.0))

    def _button(self, frame: DecodedT12Frame, name: str) -> int:
        spec = _mapping(self.channels[name], name=name)
        raw = self._raw(frame, spec)
        kind = str(spec.get("kind", "binary"))
        if kind == "binary":
            threshold = float(spec.get("threshold", 0.5 * (float(spec["min"]) + float(spec["max"]))))
            pressed = raw >= threshold
            if not bool(spec.get("pressed_high", True)):
                pressed = not pressed
            return 1 if pressed else -1
        if kind == "three_way":
            minimum = float(spec["min"])
            center = float(spec["center"])
            maximum = float(spec["max"])
            low_threshold = 0.5 * (minimum + center)
            high_threshold = 0.5 * (center + maximum)
            value = -1 if raw < low_threshold else 1 if raw > high_threshold else 0
            return -value if bool(spec.get("reversed", False)) else value
        raise ValueError(f"unsupported {name} kind: {kind}")

    def snapshot(
        self, frame: DecodedT12Frame, *, timestamp: float
    ) -> T12RemoteSnapshot:
        values = {name: self._axis(frame, name) for name in self.AXES}
        values.update({name: self._button(frame, name) for name in self.BUTTONS})
        return T12RemoteSnapshot(
            **values,
            timestamp=float(timestamp),
            connected=not frame.frame_lost and not frame.failsafe,
            failsafe=frame.failsafe,
            frame_lost=frame.frame_lost,
            raw_channels=frame.channels,
        )


def load_t12_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("T12 config root must be a mapping")
    payload["_path"] = str(config_path)
    return payload


def _normalize_channel_config(config: dict[str, Any]) -> dict[str, Any]:
    """Accept both the tutorial schema and the compact internal schema."""

    if "channels" in config:
        return _mapping(config["channels"], name="channels")
    calibration = _mapping(config.get("calibration"), name="calibration")
    axes = _mapping(calibration.get("axes"), name="calibration.axes")
    controls = _mapping(
        calibration.get("controls"), name="calibration.controls"
    )
    channel_base = int(calibration.get("channel_base", 1))
    output: dict[str, Any] = {}
    for name, raw_spec in axes.items():
        spec = _mapping(raw_spec, name=f"calibration.axes.{name}")
        output[name] = {
            "index": int(spec["channel"]) - channel_base,
            "min": spec["raw_min"],
            "center": spec["raw_center"],
            "max": spec["raw_max"],
            "reversed": bool(spec.get("reversed", False)),
        }
    for name, raw_spec in controls.items():
        spec = _mapping(raw_spec, name=f"calibration.controls.{name}")
        index = int(spec["channel"]) - channel_base
        if "pressed" in spec:
            released = float(spec["released"])
            pressed = float(spec["pressed"])
            output[name] = {
                "index": index,
                "kind": "binary",
                "min": min(released, pressed),
                "max": max(released, pressed),
                "pressed_high": pressed > released,
            }
        else:
            down = float(spec["down"])
            neutral = float(spec["neutral"])
            up = float(spec["up"])
            if not min(down, up) < neutral < max(down, up):
                raise ValueError(f"{name} neutral value must lie between down/up")
            output[name] = {
                "index": index,
                "kind": "three_way",
                "min": min(down, up),
                "center": neutral,
                "max": max(down, up),
                "reversed": up < down,
            }
    return output


class T12SerialProvider:
    """Read and decode serial data in a daemon thread; consumers never block."""

    def __init__(self, config: str | Path | dict[str, Any]) -> None:
        self.config = load_t12_config(config) if not isinstance(config, dict) else dict(config)
        transport = _mapping(
            self.config.get("transport", self.config), name="transport"
        )
        self.device = str(transport["device"])
        self.baudrate = int(transport.get("baudrate", 115200))
        self.timeout_s = float(
            transport.get("failsafe_timeout_s", self.config.get("timeout_s", 0.20))
        )
        protocol = str(
            transport.get("frame_format", self.config.get("protocol", "vendor35"))
        ).lower()
        protocol = {
            "vendor_decoded": "vendor35",
            "vendor_decoded_35": "vendor35",
            "raw_sbus_25": "sbus25",
        }.get(protocol, protocol)
        if protocol == "vendor35":
            self.decoder: VendorDecoded35Decoder | RawSbus25Decoder = VendorDecoded35Decoder(
                lost_mask=int(transport.get("lost_mask", self.config.get("lost_mask", 0))),
                failsafe_mask=int(
                    transport.get("failsafe_mask", self.config.get("failsafe_mask", 0))
                ),
            )
        elif protocol == "sbus25":
            self.decoder = RawSbus25Decoder()
        else:
            raise ValueError("T12 protocol must be vendor35 or sbus25")
        self.mapper = T12ChannelMapper(_normalize_channel_config(self.config))
        self._lock = threading.Lock()
        self._latest: T12RemoteSnapshot | None = None
        self._error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._serial: Any | None = None
        self._log_path = self.config.get("log_jsonl")
        self._log_file: Any | None = None

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def start(self) -> "T12SerialProvider":
        if self._thread is not None:
            return self
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("T12 serial input requires pyserial") from exc
        self._serial = serial.Serial(
            self.device,
            self.baudrate,
            timeout=0.05,
            exclusive=True,
        )
        if self._log_path:
            self._log_file = Path(str(self._log_path)).expanduser().open(
                "a", encoding="utf-8"
            )
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="t12-serial-reader", daemon=True
        )
        self._thread.start()
        return self

    def _record(self, snapshot: T12RemoteSnapshot) -> None:
        if self._log_file is None:
            return
        payload = dict(snapshot.__dict__)
        payload["raw_channels"] = list(snapshot.raw_channels)
        self._log_file.write(json.dumps(payload, sort_keys=True) + "\n")
        self._log_file.flush()

    def _run(self) -> None:
        assert self._serial is not None
        try:
            while not self._stop.is_set():
                data = self._serial.read(max(1, int(self._serial.in_waiting or 1)))
                if not data:
                    continue
                for frame in self.decoder.feed(data):
                    snapshot = self.mapper.snapshot(frame, timestamp=time.monotonic())
                    with self._lock:
                        self._latest = snapshot
                        self._error = None
                    self._record(snapshot)
        except Exception as exc:  # serial unplug/error must fail safe
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"

    def latest(self, *, now: float | None = None) -> T12RemoteSnapshot | None:
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            snapshot = self._latest
            error = self._error
        if snapshot is None:
            return None
        if error is not None or timestamp - snapshot.timestamp > self.timeout_s:
            return replace(snapshot, connected=False, failsafe=True)
        return snapshot

    read = latest

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._serial is not None:
            self._serial.close()
            self._serial = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __enter__(self) -> "T12SerialProvider":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()


def monitor(config: str | Path) -> None:
    with T12SerialProvider(config) as provider:
        print(f"[T12] monitoring {provider.device} at {provider.baudrate} baud")
        last_timestamp = -1.0
        window_start = time.monotonic()
        window_frames = 0
        last_print = float("-inf")
        try:
            while True:
                snapshot = provider.latest()
                if snapshot is not None and snapshot.timestamp != last_timestamp:
                    last_timestamp = snapshot.timestamp
                    window_frames += 1
                    now = time.monotonic()
                    if now - last_print >= 0.25:
                        payload = dict(snapshot.__dict__)
                        payload["raw_channels"] = list(snapshot.raw_channels)
                        payload["frame_age_s"] = max(0.0, now - snapshot.timestamp)
                        payload["frame_hz"] = window_frames / max(
                            now - window_start, 1.0e-6
                        )
                        print(json.dumps(payload, sort_keys=True))
                        last_print = now
                elif provider.error:
                    print(f"[T12] {provider.error}")
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor a T12/R12 serial link")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    monitor(args.config)


if __name__ == "__main__":
    main()


__all__ = [
    "DecodedT12Frame",
    "RawSbus25Decoder",
    "T12ChannelMapper",
    "T12RemoteSnapshot",
    "T12SerialProvider",
    "VendorDecoded35Decoder",
    "load_t12_config",
    "monitor",
]


