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

## Config

Copy `.env.example` to `.env` and fill in email creds, idle threshold, poll interval,
and power cap.

### Email (Gmail app password)

`send_email.py` / `gpu_idle_alert.sh` send over SMTP. With Gmail, enable 2-Step
Verification, then create an **App password** (Google Account → Security → App passwords —
a 16-char code) and use that, not your normal login password:

```bash
# scripts/gpu/.env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=465
SMTP_USER=you@gmail.com
SMTP_PASSWORD=abcd efgh ijkl mnop   # 16-char app password (spaces optional)
SMTP_TO=you@gmail.com
```

### GPU Idle Monitoring

`gpu_idle_alert.sh` flags a GPU as idle only when **every** sample over a window stays
at/below the threshold. Recommend **util ≤ 5% for 10 min** — a one-shot check fires false
positives on brief lulls between batches. Window = `GPU_IDLE_SAMPLES` × `GPU_IDLE_INTERVAL`
(defaults give 10 min):

```bash
# scripts/gpu/.env
GPU_IDLE_UTIL_THRESHOLD=5   # % at/below which a GPU counts as idle
GPU_IDLE_SAMPLES=60         # all samples must be idle
GPU_IDLE_INTERVAL=10        # seconds between samples → 60 × 10s = 10 min
```

### Power Control and Cooling

install dependency use following command:
```bash
sudo apt install -y python3-pynvml
```

run with following command

```bash
sudo python3 scripts/gpu/gpu_fans.py 55
sudo python3 scripts/gpu/gpu_power.py 400 --gpu 0
```

Recommended settings (RTX 6000 Pro / RTX 5090)  
- **Power cap 400W**: only ~10% performance loss, but runs much safer and cooler.
  Strongly recommended over the stock limit for sustained loads.
- **Fan speed 55%**: tested to keep the GPU under 70°C at ~28°C ambient
  (typical Bay Area room temperature).