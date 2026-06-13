# gpu

GPU environment check and monitoring utilities.

## Scripts

| Script             | Purpose                                                       |
|--------------------|---------------------------------------------------------------|
| `check_env.py`     | torch/CUDA/cuDNN/NCCL + per-GPU name & compute capability      |
| `gpu_fans.py`      | fan speed monitor / control                                   |
| `gpu_power.py`     | power draw monitor / power-cap setter                         |
| `gpu_idle_alert.sh`| detect idle GPU(s) and alert via `send_email.py`              |
| `send_email.py`    | shared email notification helper                              |
| `setup_crontab.sh` | install idle-alert / monitors as cron jobs                   |

## Config

Copy `.env.example` to `.env` and fill in email creds, idle threshold, poll interval,
and power cap.

`gpu_power.py` reads `GPU_POWER_CAP_W` from `.env` when no watts are passed:

```bash
sudo -E python3 gpu_power.py            # apply GPU_POWER_CAP_W from .env to all GPUs
sudo python3 gpu_power.py 300 --gpu 0   # explicit value still overrides
```

(`sudo -E` preserves your environment; without it, set the cap on the CLI instead.)

<!-- TODO: fill scripts -->
