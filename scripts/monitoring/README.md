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
cd prometheus && sudo docker compose up -d && cd ..
cd node-exporter && sudo docker compose up -d && cd ..
cd nvidia-smi-exporter && sudo docker compose up -d && cd ..
cd grafana && sudo docker compose up -d && cd ..
```

## Endpoints

- Prometheus: http://localhost:9090
- node-exporter: http://localhost:9100/metrics
- nvidia_smi_exporter: http://localhost:9835/metrics
- Grafana: http://localhost:3000

In Grafana, add Prometheus as a datasource at `http://localhost:9090`. Recommended
dashboards: node-exporter "Node Exporter Full" (ID 1860), and the nvidia_gpu_exporter
dashboard (ID 14574).

## Grafana Dashboards
- [linux metrics](https://grafana.com/grafana/dashboards/1860-node-exporter-full/)
- [nvidia-smi metrics](https://grafana.com/grafana/dashboards/12357-nvidia-smi-graphs/)