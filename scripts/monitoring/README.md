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

## Bring-up

```bash
cd prometheus && docker compose up -d && cd -
cd node-exporter && docker compose up -d && cd -
cd nvidia-smi-exporter && docker compose up -d && cd -   # needs NVIDIA docker runtime
cd grafana && docker compose up -d && cd -
```

## Endpoints

- Prometheus: http://localhost:9090
- node-exporter: http://localhost:9100/metrics
- nvidia_smi_exporter: http://localhost:9835/metrics
- Grafana: http://localhost:3000 (add Prometheus `http://localhost:9090` as datasource)

## Cloudflare Tunnel

<!-- TODO: cloudflared tunnel setup + ingress map (user-provided token) -->

<!-- TODO: example PromQL (GPU util/power/temp, host load) -->
