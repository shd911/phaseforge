#!/bin/zsh
# MLX Local Models — startup script
# Run from any directory

MODEL_QWEN="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit"
MODEL_GEMMA="mlx-community/gemma-4-31b-it-4bit"
PORT=10240

echo "Starting MLX server on port $PORT..."
echo "Open WebUI connection: http://localhost:$PORT/v1  (API key: mlx)"
echo ""
echo "Models available:"
echo "  - $MODEL_QWEN"
echo "  - $MODEL_GEMMA (only if fully downloaded)"
echo ""
echo "Press Ctrl+C to stop."
echo ""

mlx_lm.server \
  --model "$MODEL_QWEN" \
  --port $PORT
