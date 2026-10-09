#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# docker-entrypoint.sh
# Принимает: --image PATH --weights PATH --config PATH [--output-dir PATH]
# ============================================================

IMAGE=""
WEIGHTS=""
CONFIG=""
OUTPUT_DIR="/app/results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)       IMAGE="$2";      shift 2 ;;
        --weights)     WEIGHTS="$2";    shift 2 ;;
        --config)      CONFIG="$2";     shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        *)             echo "Неизвестный аргумент: $1"; exit 1 ;;
    esac
done

if [[ -z "$IMAGE" || -z "$WEIGHTS" || -z "$CONFIG" ]]; then
    echo "Использование: $0 --image PATH --weights PATH --config PATH [--output-dir PATH]"
    echo "Пример:"
    echo "  docker run --rm -v \$(pwd)/data:/data -v \$(pwd)/results:/results ai_gen:cpu \\"
    echo "    --image /data/input.png --weights /data/best.pt --config /data/config.yaml"
    exit 1
fi

for p in "$IMAGE" "$WEIGHTS" "$CONFIG"; do
    if [[ ! -e "$p" ]]; then
        echo "❌ Файл не найден внутри контейнера: $p"
        echo "   Убедитесь, что путь смонтирован через -v."
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"

echo "🚀 Запуск конвейера..."
echo "   image:   $IMAGE"
echo "   weights: $WEIGHTS"
echo "   config:  $CONFIG"
echo "   output:  $OUTPUT_DIR"

exec python full_pipeline_v2.py \
    --config "$CONFIG" \
    --image "$IMAGE" \
    --weights "$WEIGHTS" \
    --output-dir "$OUTPUT_DIR"