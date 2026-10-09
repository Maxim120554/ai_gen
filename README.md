# AI Gen Pipeline (Docker)

Конвейер генерализации карт на основе YOLO и HDBSCAN.

## 🚀 Запуск

### Linux / macOS
```bash
docker compose run --rm \
  -v ./input.png:/data/input.png \
  -v ./best.pt:/data/best.pt \
  -v ./config.yaml:/data/config.yaml \
  ai_gen \
  --image /data/input.png \
  --weights /data/best.pt \
  --config /data/config.yaml \
  --output-dir /results
```

### 🪟 Windows (PowerShell)
```bash
docker compose run --rm `
  -v .\input.png:/data/input.png `
  -v .\best.pt:/data/best.pt `
  -v .\config.yaml:/data/config.yaml `
  ai_gen `
  --image /data/input.png `
  --weights /data/best.pt `
  --config /data/config.yaml `
  --output-dir /results
```