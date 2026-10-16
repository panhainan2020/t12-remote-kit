"""Interactive, read-only detector for a physical T12 serial channel map.

This tool deliberately works from the receiver's raw 16-channel frames. It
does not trust the current YAML field names, start MuJoCo, or send any motion
command. The operator moves one physical control at a time and the detector
records which raw channel actually changed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Iterable

import numpy as np

from .serial import T12SerialProvider


@dataclass(frozen=True)
class DetectionStep:
    key: str
    control: str
    position: str
    prompt: str


@dataclass(frozen=True)
class ChannelDetection:
    channel: int
    baseline_raw: int
    active_raw: int
    delta: int


BUTTON_STEPS = (
    DetectionStep("button_e_outer", "button_e", "outer", "将 E 拨到外侧（远离遥控器中心）并保持"),
    DetectionStep("button_e_inner", "button_e", "inner", "将 E 拨到内侧（朝向遥控器中心）并保持"),
    DetectionStep("button_f_outer", "button_f", "outer", "将 F 拨到外侧（远离遥控器中心）并保持"),
    DetectionStep("button_f_inner", "button_f", "inner", "将 F 拨到内侧（朝向遥控器中心）并保持"),
    DetectionStep("button_g_outer", "button_g", "outer", "将 G 拨到外侧（远离遥控器中心）并保持"),
    DetectionStep("button_g_inner", "button_g", "inner", "将 G 拨到内侧（朝向遥控器中心）并保持"),
    DetectionStep("button_h_outer", "button_h", "outer", "将 H 拨到外侧（远离遥控器中心）并保持"),
    DetectionStep("button_h_inner", "button_h", "inner", "将 H 拨到内侧（朝向遥控器中心）并保持"),
    DetectionStep("button_a_pressed", "button_a", "pressed", "按住 A"),
    DetectionStep("button_b_pressed", "button_b", "pressed", "按住 B"),
    DetectionStep("button_c_pressed", "button_c", "pressed", "按住 C"),
    DetectionStep("button_d_pressed", "button_d", "pressed", "按住 D"),
)

SWITCH_STEPS = BUTTON_STEPS[:8]
PUSHBUTTON_STEPS = BUTTON_STEPS[8:]


AXIS_STEPS = (
    DetectionStep("y1_positive", "y1", "positive", "右摇杆 Y1 向前推到底并保持"),
    DetectionStep("y1_negative", "y1", "negative", "右摇杆 Y1 向后推到底并保持"),
    DetectionStep("x1_positive", "x1", "positive", "右摇杆 X1 向右推到底并保持"),
    DetectionStep("x1_negative", "x1", "negative", "右摇杆 X1 向左推到底并保持"),
    DetectionStep("x2_positive", "x2", "positive", "左摇杆 X2 向右推到底并保持"),
    DetectionStep("x2_negative", "x2", "negative", "左摇杆 X2 向左推到底并保持"),
    DetectionStep("y2_positive", "y2", "positive", "左摇杆 Y2 向前推到底并保持"),
    DetectionStep("y2_negative", "y2", "negative", "左摇杆 Y2 向后推到底并保持"),
)


def detect_changed_channel(
    baseline: Iterable[int],
    active: Iterable[int],
    *,
    threshold: int = 180,
) -> ChannelDetection:
    """Return the single raw channel that moved materially from baseline."""

    baseline_values = tuple(int(value) for value in baseline)
    active_values = tuple(int(value) for value in active)
    if len(baseline_values) != len(active_values):
        raise ValueError("baseline and active channel counts differ")
    changed = [
        index
        for index, (center, value) in enumerate(zip(baseline_values, active_values))
        if abs(value - center) >= threshold
    ]
    if not changed:
        raise ValueError("没有通道达到变化阈值")
    if len(changed) != 1:
        labels = ", ".join(f"CH{index + 1}" for index in changed)
        raise ValueError(f"同时检测到多个变化通道：{labels}")
    index = changed[0]
    return ChannelDetection(
        channel=index + 1,
        baseline_raw=baseline_values[index],
        active_raw=active_values[index],
        delta=active_values[index] - baseline_values[index],
    )


def _sample_median(
    provider: T12SerialProvider,
    *,
    duration_s: float,
    minimum_frames: int = 5,
) -> tuple[int, ...]:
    deadline = time.monotonic() + duration_s
    last_timestamp = -1.0
    samples: list[tuple[int, ...]] = []
    while time.monotonic() < deadline:
        snapshot = provider.latest()
        if (
            snapshot is not None
            and snapshot.connected
            and not snapshot.failsafe
            and snapshot.timestamp != last_timestamp
        ):
            last_timestamp = snapshot.timestamp
            samples.append(tuple(int(value) for value in snapshot.raw_channels))
        time.sleep(0.005)
    if len(samples) < minimum_frames:
        detail = provider.error or f"只收到 {len(samples)} 帧"
        raise RuntimeError(f"无法获得稳定的 T12 原始数据：{detail}")
    widths = {len(sample) for sample in samples}
    if len(widths) != 1:
        raise RuntimeError("采样期间 T12 通道数量发生变化")
    return tuple(int(round(value)) for value in np.median(samples, axis=0))


def _wait_for_change(
    provider: T12SerialProvider,
    baseline: tuple[int, ...],
    *,
    threshold: int,
    stable_s: float,
    timeout_s: float,
) -> ChannelDetection:
    deadline = time.monotonic() + timeout_s
    candidate: ChannelDetection | None = None
    candidate_since = 0.0
    last_notice = ""
    while time.monotonic() < deadline:
        snapshot = provider.latest()
        if snapshot is None or not snapshot.connected or snapshot.failsafe:
            time.sleep(0.01)
            continue
        try:
            current = detect_changed_channel(
                baseline, snapshot.raw_channels, threshold=threshold
            )
        except ValueError as error:
            notice = str(error)
            if "多个" in notice and notice != last_notice:
                print(f"  [等待] {notice}；请只操作当前提示的一个控件", flush=True)
                last_notice = notice
            candidate = None
            time.sleep(0.01)
            continue
        now = time.monotonic()
        same_candidate = (
            candidate is not None
            and candidate.channel == current.channel
            and np.sign(candidate.delta) == np.sign(current.delta)
        )
        if not same_candidate:
            candidate = current
            candidate_since = now
        elif now - candidate_since >= stable_s:
            return current
        time.sleep(0.01)
    raise TimeoutError("等待控件动作超时")


def _wait_for_neutral(
    provider: T12SerialProvider,
    baseline: tuple[int, ...],
    *,
    tolerance: int,
    stable_s: float,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    neutral_since: float | None = None
    while time.monotonic() < deadline:
        snapshot = provider.latest()
        if snapshot is None or not snapshot.connected or snapshot.failsafe:
            neutral_since = None
            time.sleep(0.01)
            continue
        neutral = all(
            abs(int(value) - center) <= tolerance
            for center, value in zip(baseline, snapshot.raw_channels)
        )
        now = time.monotonic()
        if neutral:
            neutral_since = now if neutral_since is None else neutral_since
            if now - neutral_since >= stable_s:
                return
        else:
            neutral_since = None
        time.sleep(0.01)
    raise TimeoutError("等待全部控件回到中位/释放状态超时")


def build_report(
    baseline: Iterable[int],
    captures: dict[str, ChannelDetection],
) -> dict[str, object]:
    """Build conflict checks and YAML-ready calibration suggestions."""

    baseline_values = tuple(int(value) for value in baseline)
    errors: list[str] = []
    suggestions: dict[str, dict[str, object]] = {}
    assigned_channels: dict[int, str] = {}

    for name in ("button_e", "button_f", "button_g", "button_h"):
        outer = captures.get(f"{name}_outer")
        inner = captures.get(f"{name}_inner")
        if outer is None or inner is None:
            continue
        if outer.channel != inner.channel:
            errors.append(
                f"{name} 的外侧/内侧落在不同通道：CH{outer.channel}/CH{inner.channel}"
            )
            continue
        channel = outer.channel
        if channel in assigned_channels:
            errors.append(f"{name} 与 {assigned_channels[channel]} 都使用 CH{channel}")
        assigned_channels[channel] = name
        suggestions[name] = {
            "channel": channel,
            "down": outer.active_raw,
            "neutral": baseline_values[channel - 1],
            "up": inner.active_raw,
        }

    for name in ("button_a", "button_b", "button_c", "button_d"):
        pressed = captures.get(f"{name}_pressed")
        if pressed is None:
            continue
        channel = pressed.channel
        if channel in assigned_channels:
            errors.append(f"{name} 与 {assigned_channels[channel]} 都使用 CH{channel}")
        assigned_channels[channel] = name
        suggestions[name] = {
            "channel": channel,
            "released": baseline_values[channel - 1],
            "pressed": pressed.active_raw,
        }

    for name in ("x1", "y1", "x2", "y2"):
        positive = captures.get(f"{name}_positive")
        negative = captures.get(f"{name}_negative")
        if positive is None or negative is None:
            continue
        if positive.channel != negative.channel:
            errors.append(
                f"{name} 的正/负方向落在不同通道：CH{positive.channel}/CH{negative.channel}"
            )
            continue
        channel = positive.channel
        if channel in assigned_channels:
            errors.append(f"{name} 与 {assigned_channels[channel]} 都使用 CH{channel}")
        assigned_channels[channel] = name
        suggestions[name] = {
            "channel": channel,
            "raw_min": min(negative.active_raw, positive.active_raw),
            "raw_center": baseline_values[channel - 1],
            "raw_max": max(negative.active_raw, positive.active_raw),
            "reversed": positive.active_raw < negative.active_raw,
        }

    return {
        "baseline_raw_channels": list(baseline_values),
        "captures": {key: asdict(value) for key, value in captures.items()},
        "suggested_calibration": suggestions,
        "errors": errors,
    }


def _yaml_lines(report: dict[str, object]) -> list[str]:
    suggestions = report["suggested_calibration"]
    assert isinstance(suggestions, dict)
    lines = ["calibration:", "  channel_base: 1", "  axes:"]
    for name in ("x1", "y1", "x2", "y2"):
        spec = suggestions.get(name)
        if isinstance(spec, dict):
            reversed_text = str(bool(spec["reversed"])).lower()
            lines.append(
                f"    {name}: {{channel: {spec['channel']}, raw_min: {spec['raw_min']}, "
                f"raw_center: {spec['raw_center']}, raw_max: {spec['raw_max']}, "
                f"reversed: {reversed_text}}}"
            )
    lines.append("  controls:")
    for name in (
        "button_e", "button_f", "button_g", "button_h",
        "button_a", "button_b", "button_c", "button_d",
    ):
        spec = suggestions.get(name)
        if not isinstance(spec, dict):
            continue
        fields = ", ".join(f"{key}: {value}" for key, value in spec.items())
        lines.append(f"    {name}: {{{fields}}}")
    return lines


def _write_report(
    output: Path,
    config: Path,
    baseline: tuple[int, ...],
    captures: dict[str, ChannelDetection],
    *,
    complete: bool,
) -> dict[str, object]:
    report = build_report(baseline, captures)
    report["generated_at"] = datetime.now().astimezone().isoformat()
    report["config"] = str(config.resolve())
    report["complete"] = complete
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def run_detector(
    config: Path,
    *,
    switch_names: tuple[str, ...] = ("e", "f", "g", "h"),
    include_pushbuttons: bool,
    include_axes: bool,
    threshold: int,
    timeout_s: float,
    output_dir: Path,
) -> Path:
    selected_switches = tuple(
        step
        for step in SWITCH_STEPS
        if step.control.removeprefix("button_") in switch_names
    )
    steps = (
        selected_switches
        + (PUSHBUTTON_STEPS if include_pushbuttons else ())
        + (AXIS_STEPS if include_axes else ())
    )
    print("[T12-DETECT] 只读映射检测：不会启动 MuJoCo，也不会发送机器人命令。")
    print("[T12-DETECT] 请先关闭所有正在使用 T12 串口的仿真进程。")
    print("[T12-DETECT] 外侧=远离遥控器中心；内侧=朝向遥控器中心。")
    input("将 E/F/G/H 全部置中、松开 A/B/C/D、摇杆回中，然后按回车：")

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"t12_mapping_report_{datetime.now():%Y%m%d_%H%M%S}.json"
    captures: dict[str, ChannelDetection] = {}
    with T12SerialProvider(config) as provider:
        baseline = _sample_median(provider, duration_s=0.6)
        print(f"[基线] CH1..CH16 = {list(baseline)}", flush=True)
        _write_report(output, config, baseline, captures, complete=False)
        print(f"[检查点] {output.resolve()}", flush=True)
        for index, step in enumerate(steps, start=1):
            print(f"\n[{index}/{len(steps)}] {step.prompt}；检测成功前不要松开。", flush=True)
            detection = _wait_for_change(
                provider,
                baseline,
                threshold=threshold,
                stable_s=0.15,
                timeout_s=timeout_s,
            )
            captures[step.key] = detection
            _write_report(output, config, baseline, captures, complete=False)
            sign = "+" if detection.delta >= 0 else ""
            print(
                f"  [捕获] {step.control} {step.position} -> CH{detection.channel}: "
                f"{detection.baseline_raw} -> {detection.active_raw} "
                f"(delta={sign}{detection.delta})",
                flush=True,
            )
            print("  请松开按钮或把拨杆/摇杆恢复中位……", flush=True)
            _wait_for_neutral(
                provider,
                baseline,
                tolerance=max(60, threshold // 3),
                stable_s=0.15,
                timeout_s=timeout_s,
            )
            print("  [回中] 已确认", flush=True)

    report = _write_report(output, config, baseline, captures, complete=True)

    print("\n[T12-DETECT] 检测完成。建议映射：")
    print("\n".join(_yaml_lines(report)))
    errors = report["errors"]
    if errors:
        print("\n[T12-DETECT] 检测到冲突，暂时不要修改正式配置：")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n[T12-DETECT] 未发现控件通道冲突。")
    print(f"[T12-DETECT] 完整报告：{output.resolve()}")
    return output


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="交互式检测 T12 物理控件对应的串口原始通道"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--switches",
        nargs="+",
        choices=("e", "f", "g", "h"),
        default=("e", "f", "g", "h"),
        help="只检测列出的三段拨杆；例如 --switches f g h",
    )
    parser.add_argument(
        "--switches-only",
        action="store_true",
        help="只检测 E/F/G/H，跳过已经确认正确的 A/B/C/D",
    )
    parser.add_argument(
        "--include-axes",
        action="store_true",
        help="在 E-H 和 A-D 之后继续检测四个摇杆轴的正负方向",
    )
    parser.add_argument("--threshold", type=int, default=180)
    parser.add_argument("--step-timeout", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    if args.threshold <= 0:
        parser.error("--threshold must be positive")
    if args.step_timeout <= 0.0:
        parser.error("--step-timeout must be positive")
    return args


def main() -> None:
    args = _parse_args()
    try:
        run_detector(
            args.config,
            switch_names=tuple(args.switches),
            include_pushbuttons=not args.switches_only,
            include_axes=args.include_axes,
            threshold=args.threshold,
            timeout_s=args.step_timeout,
            output_dir=args.output_dir,
        )
    except (KeyboardInterrupt, EOFError):
        print("\n[T12-DETECT] 已取消；正式映射配置没有被修改。")
    except (OSError, RuntimeError, TimeoutError) as error:
        print(f"\n[T12-DETECT] 错误：{error}")
        print(
            "[T12-DETECT] 若串口被占用，请先在运行 MuJoCo/monitor 的终端按 Ctrl+C，"
            "然后重新执行检测命令。"
        )
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()


__all__ = [
    "AXIS_STEPS",
    "BUTTON_STEPS",
    "PUSHBUTTON_STEPS",
    "SWITCH_STEPS",
    "ChannelDetection",
    "DetectionStep",
    "build_report",
    "detect_changed_channel",
    "run_detector",
]

