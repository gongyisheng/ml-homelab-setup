# monitoring

Prometheus metrics stack: Prometheus + node-exporter + GPU exporter + Grafana, exposed
remotely via Cloudflare Tunnel. Each component is its own Docker Compose file (host
networking).

## Layout

```
monitoring/
├── prometheus/          docker-compose.yml + prometheus.yml   (:9090)
├── node-exporter/       docker-compose.yml                    (:9100)
├── nvidia-smi-exporter/ docker-compose.yml                    (:9835)
└── grafana/             docker-compose.yml                    (:3000)
```

The GPU exporter (`utkuozdemir/nvidia_gpu_exporter`) needs the NVIDIA Docker runtime from
`../bootstrap/install_cuda_container_kit.sh`.

## Bring-up

```bash
cd prometheus && docker compose up -d && cd -
cd node-exporter && docker compose up -d && cd -
cd nvidia-smi-exporter && docker compose up -d && cd -
cd grafana && docker compose up -d && cd -
```

## Endpoints

- Prometheus: http://localhost:9090
- node-exporter: http://localhost:9100/metrics
- nvidia_smi_exporter: http://localhost:9835/metrics
- Grafana: http://localhost:3000

In Grafana, add Prometheus as a datasource at `http://localhost:9090`. Recommended
dashboards: node-exporter "Node Exporter Full" (ID 1860), and the nvidia_gpu_exporter
dashboard (ID 14574).

## Example PromQL

```promql
# Per-GPU power draw (W)
nvidia_smi_power_draw_instant_watts

# Per-GPU utilization (%)
nvidia_smi_utilization_gpu_ratio * 100

# Per-GPU temperature (C)
nvidia_smi_temperature_gpu

# Host CPU busy (%)
100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Host memory used (%)
(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100
```

## Cloudflare Tunnel

Expose Grafana (and exporters if desired) remotely without opening inbound ports. The
tunnel token is a **credential** — keep it out of git (store in `cloudflared/.env` or the
host's `/etc/cloudflared/`).

One-time setup (token-based, managed from the Cloudflare Zero Trust dashboard):

```bash
# Run the connector as a container, token from the dashboard tunnel.
docker run -d --restart always --name cloudflared --network host \
  cloudflare/cloudflared:latest tunnel --no-autoupdate run --token <TUNNEL_TOKEN>
```

Then in the Cloudflare dashboard add public hostnames (ingress) routing to local services,
e.g.:

| Public hostname             | Service                  |
|-----------------------------|--------------------------|
| grafana.example.com         | http://localhost:3000    |
| prometheus.example.com      | http://localhost:9090    |

For a config-file (non-token) tunnel, the equivalent ingress lives in
`~/.cloudflared/config.yml`:

```yaml
tunnel: <TUNNEL_ID>
credentials-file: /root/.cloudflared/<TUNNEL_ID>.json
ingress:
  - hostname: grafana.example.com
    service: http://localhost:3000
  - service: http_status:404
```

<!-- Paste real tunnel token / ingress here as source of truth; do not commit secrets. -->
