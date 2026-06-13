# gpu

GPU monitoring utilities. (Env diagnostic `check_env.py` lives in `../bootstrap/`.)

## Scripts

| Script             | Purpose                                                       |
|--------------------|---------------------------------------------------------------|
| `gpu_fans.py`      | fan speed monitor / control                                   |
| `gpu_power.py`     | power draw monitor / power-cap setter                         |
| `gpu_idle_alert.sh`| detect idle GPU(s) and alert via `send_email.py`              |
| `send_email.py`    | shared email notification helper                              |
| `setup_crontab.sh` | install idle-alert / monitors as cron jobs                   |

## Run

Fan/power control need root, so call the venv's python directly under sudo
(`nvidia-ml-py` lives in `.venv`). Paths below are from the repo root:

```bash
sudo .venv/bin/python scripts/gpu/gpu_fans.py 70
```

## Config

Copy `.env.example` to `.env` and fill in email creds, idle threshold, poll interval,
and power cap.

`gpu_power.py` reads `GPU_POWER_CAP_W` from `.env` when no watts are passed:

```bash
sudo -E .venv/bin/python scripts/gpu/gpu_power.py          # apply GPU_POWER_CAP_W to all GPUs
sudo .venv/bin/python scripts/gpu/gpu_power.py 300 --gpu 0  # explicit value still overrides
```

(`sudo -E` preserves your environment; without it, set the cap on the CLI instead.)
