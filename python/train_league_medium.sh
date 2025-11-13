#!/bin/bash
# Medium-length league training script with all optimizations
# Expected duration: 48-72 hours on modern GPU
# GPU memory requirement: 8GB+ recommended

# Activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate torch

# Configuration
SAVE_DIR="league_medium_$(date +%Y%m%d_%H%M%S)"
ITERATIONS=500
MAIN_GAMES=20
EXPLOITER_GAMES=30
EVAL_INTERVAL=25
EVAL_GAMES=30
EVAL_DEPTHS="2 3 4"
HIDDEN_DIM=32
NUM_LAYERS=3
INFERENCE_BATCH_SIZE=512

echo "=================================================="
echo "Starting Medium-Length League Training"
echo "=================================================="
echo "Save directory: $SAVE_DIR"
echo "Total iterations: $ITERATIONS"
echo "Estimated duration: 48-72 hours"
echo "=================================================="
echo ""

# Run training with all optimizations enabled
python train_league.py \
    --iterations $ITERATIONS \
    --main-games $MAIN_GAMES \
    --exploiter-games $EXPLOITER_GAMES \
    --hidden-dim $HIDDEN_DIM \
    --num-layers $NUM_LAYERS \
    --save-dir "$SAVE_DIR" \
    --eval-interval $EVAL_INTERVAL \
    --eval-games $EVAL_GAMES \
    --eval-depths $EVAL_DEPTHS \
    --use-amp \
    --use-compile \
    --inference-batch-size $INFERENCE_BATCH_SIZE \
    2>&1 | tee "${SAVE_DIR}_training.log"

echo ""
echo "=================================================="
echo "Training completed!"
echo "Results saved to: $SAVE_DIR"
echo "Log file: ${SAVE_DIR}_training.log"
echo "=================================================="
