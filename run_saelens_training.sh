#!/bin/bash

# SAELens v6 Training Script
# Successfully tested and working configuration
# Model: meta-llama/Meta-Llama-3-8B, Layer 12, 500k tokens

echo "🚀 SAELens v6 Training Script"
echo "📋 Configuration:"
echo "   Model: meta-llama/Meta-Llama-3-8B"
echo "   Layer: 12"
echo "   Training tokens: 500,000"
echo "   Normalization: none"
echo "   Dataset: jyanimaulik/yahoo_finance_stockmarket_news"
echo "   Batch size: 1024"
echo "   Context size: 2048"
echo "   Expansion factor: 32"
echo "   L1 coefficient: 5.0"
echo "   Learning rate: 1e-4"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

# Run SAELens training
cd /home/nvidia/Documents/Hariom/SAELensTrain

echo "🎯 Starting SAELens training..."
echo "📁 Using SAELens training script: sae_lens_train.py"
./run_simple.sh 1 "0" \
    --model_name "meta-llama/Meta-Llama-3-8B" \
    --layer 12 \
    --expansion_factor 32 \
    --training_tokens 500000 \
    --lr 1e-4 \
    --l1_coefficient 5.0 \
    --normalize_activations none \
    --output_name "llama_layer12_sae_saelens_500k" \
    --dataset_path "jyanimaulik/yahoo_finance_stockmarket_news" \
    --train_batch_size_tokens 1024 \
    --context_size 2048

echo "✅ SAELens training completed!"
