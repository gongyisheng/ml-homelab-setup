#!/usr/bin/env python3
"""
Set power limit on NVIDIA GPUs via NVML. Requires root.

sudo apt install -y python3-pynvml
sudo python3 gpu_power.py 300           # all GPUs, 300W
sudo python3 gpu_power.py 300 --gpu 0   # GPU 0 only
sudo python3 gpu_power.py 300 --gpu 0,1 # GPUs 0 and 1
sudo python3 gpu_power.py default       # restore driver default
sudo -E python3 gpu_power.py            # use GPU_POWER_CAP_W from .env / env

Watts defaults to $GPU_POWER_CAP_W (from scripts/gpu/.env) when not given on the CLI.
"""
import argparse
import os
import sys
from pathlib import Path
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlDeviceGetPowerManagementLimit,
    nvmlDeviceGetPowerManagementDefaultLimit,
    nvmlDeviceGetPowerManagementLimitConstraints,
    nvmlDeviceSetPowerManagementLimit,
    NVMLError,
)


def parse_watts(arg: str) -> int | None:
    arg = arg.lower()
    if arg == "default":
        return None
    value = int(arg)
    if value <= 0:
        raise ValueError("power limit must be a positive integer (watts)")
    return value


def load_dotenv(path: Path) -> None:
    """Populate os.environ from a KEY=VALUE file. Existing env vars take precedence."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def parse_gpus(arg: str | None, total: int) -> list[int]:
    if arg is None:
        return list(range(total))
    indices = [int(x) for x in arg.split(",") if x.strip()]
    for i in indices:
        if not 0 <= i < total:
            raise ValueError(f"GPU index {i} out of range (have {total} GPUs)")
    return indices


def main() -> int:
    parser = argparse.ArgumentParser(description="Set NVIDIA GPU power limit via NVML.")
    parser.add_argument(
        "watts", nargs="?",
        help="power limit in watts (e.g. 300), or 'default'. Falls back to $GPU_POWER_CAP_W.",
    )
    parser.add_argument(
        "--gpu",
        help="comma-separated GPU indices (e.g. 0 or 0,1). Defaults to all GPUs.",
    )
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent / ".env")
    watts = args.watts if args.watts is not None else os.environ.get("GPU_POWER_CAP_W")
    if watts is None:
        parser.error("no watts given and GPU_POWER_CAP_W not set (CLI arg or scripts/gpu/.env)")

    try:
        target_w = parse_watts(watts)
    except ValueError as e:
        parser.error(str(e))

    nvmlInit()
    try:
        gpus = parse_gpus(args.gpu, nvmlDeviceGetCount())
        for i in gpus:
            h = nvmlDeviceGetHandleByIndex(i)
            name = nvmlDeviceGetName(h)
            before_mw = nvmlDeviceGetPowerManagementLimit(h)
            default_mw = nvmlDeviceGetPowerManagementDefaultLimit(h)
            min_mw, max_mw = nvmlDeviceGetPowerManagementLimitConstraints(h)

            apply_mw = default_mw if target_w is None else target_w * 1000
            tag = "default" if target_w is None else f"{target_w}W"

            if apply_mw < min_mw or apply_mw > max_mw:
                print(
                    f"GPU {i} ({name}): {target_w}W out of range "
                    f"[{min_mw // 1000}W, {max_mw // 1000}W]",
                    file=sys.stderr,
                )
                continue

            try:
                nvmlDeviceSetPowerManagementLimit(h, apply_mw)
            except NVMLError as e:
                print(f"GPU {i} ({name}): FAILED -- {e}", file=sys.stderr)
                continue

            after_mw = nvmlDeviceGetPowerManagementLimit(h)
            print(
                f"GPU {i} ({name}): {before_mw // 1000}W -> {after_mw // 1000}W (target {tag})"
            )
    finally:
        nvmlShutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
