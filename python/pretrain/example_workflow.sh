#!/bin/bash
# Example workflow for pre-training and then self-play training

# Step 1: Pre-train the model using evaluation matching
# This will generate ~1000 games with the engine at depth 7
# and train for 50 epochs to match the analytical evaluation
echo "Step 1: Pre-training..."
python train.py \
    --pretrain eval-matching \
    --pretrain-games 1000 \
    --pretrain-depth 7 \
    --pretrain-epochs 50 \
    --pretrain-randomness 0.15 \
    --batch-size 64 \
    --lr 1e-3 \
    --model-path checkpoints_pretrained

# The above command will:
# - Generate training data (saved to checkpoints_pretrained/pretrain_eval_d7_g1000_r15.pkl)
# - Train the model for 50 epochs
# - Evaluate against engine to get initial ELO
# - Save the pre-trained model to checkpoints_pretrained/model_pretrained.pt
# - Exit (does not continue to self-play)

echo ""
echo "Step 2: Self-play training from pre-trained model..."
# Step 2: Continue with self-play training from the pre-trained checkpoint
python train.py \
    --resume checkpoints_pretrained/model_pretrained.pt \
    --iterations 100 \
    --games 100 \
    --epochs 10 \
    --use-td \
    --gamma 0.99 \
    --temperature 1.0 \
    --batch-size 32 \
    --lr 5e-4 \
    --eval-interval 5 \
    --eval-games 20 \
    --eval-depths 3 5 \
    --model-path checkpoints_pretrained

# Alternative: If you want to re-train with existing data (e.g., more epochs)
# python train.py \
#     --pretrain eval-matching \
#     --pretrain-data-path checkpoints_pretrained/pretrain_eval_d7_g1000_r15.pkl \
#     --pretrain-epochs 100 \
#     --model-path checkpoints_pretrained
