#!/bin/bash
# Quick start script for league training

# Activate virtual environment
source .venv/bin/activate

# Run smoke tests
echo "Running system tests..."
python test_league.py

if [ $? -eq 0 ]; then
    echo ""
    echo "Tests passed! Starting league training..."
    echo ""
    
    # Start a small training run
    python train_league.py \
        --iterations 100 \
        --main-games 5 \
        --exploiter-games 10 \
        --hidden-dim 64 \
        --num-layers 3 \
        --eval-interval 20 \
        --save-dir league_demo
    
    echo ""
    echo "Training complete! View results:"
    echo "  tensorboard --logdir league_demo/logs"
else
    echo "Tests failed. Please fix errors before training."
    exit 1
fi
