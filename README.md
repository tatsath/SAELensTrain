# SAE Training and Evaluation with SAELens

This repository contains the implementation and evaluation of Sparse Autoencoders (SAEs) using the SAELens library, trained on Meta-Llama-3-8B model activations.

## 📋 Overview

This project demonstrates:
- Training SAEs using SAELens v6.6.4
- Evaluating SAEs using real model activations
- Comparing different SAE libraries and methodologies
- Comprehensive health assessment using SAEBench metrics

## 🚀 Quick Start

### Prerequisites

```bash
# Install conda environment
conda create -n sae python=3.12
conda activate sae

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets safetensors wandb
pip install sae-lens --upgrade  # SAELens v6.6.4
pip install scikit-learn scipy numpy tqdm
```

### Authentication

```bash
# Login to HuggingFace and Weights & Biases
huggingface-cli login
wandb login
```

## 🏗️ Project Structure

```
SAELensTrain/
├── sae_lens_train.py              # SAELens v6 training script
├── run_saelens_training.sh        # Training execution script
├── evaluate_sae_real_activations.py # SAE evaluation script
├── checkpoints/                   # Trained SAE checkpoints
│   └── llama_layer12_sae_saelens_500k_8d0fb52d/
│       └── y05st1pp/final_500736/
│           ├── cfg.json
│           ├── runner_cfg.json
│           ├── sae_weights.safetensors
│           └── sparsity.safetensors
└── README.md                      # This file
```

## 🎯 Training Configuration

### SAELens v6 Training Parameters

```bash
# Model and Layer
Model: meta-llama/Meta-Llama-3-8B
Layer: 12 (hook_resid_post)
Dataset: jyanimaulik/yahoo_finance_stockmarket_news

# SAE Architecture
Expansion Factor: 32x (131,072 features)
Input Dimension: 4,096
SAE Dimension: 131,072

# Training Parameters
Training Tokens: 500,000
Learning Rate: 1e-4
L1 Coefficient: 5.0
Normalization: none
Batch Size: 1,024 tokens
Context Size: 2,048
Buffer Batches: 64 (default)

# Hardware
Device: CUDA (1 GPU)
```

### Training Command

```bash
# Execute training
./run_saelens_training.sh

# Or run manually:
cd /home/nvidia/Documents/Hariom/SAELensTrain
conda activate sae

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
```

## 📊 Evaluation Results

### SAE Health Assessment (Real Model Activations)

**Model**: meta-llama/Meta-Llama-3-8B (Layer 12)  
**SAE**: 131,072 features (32x expansion)  
**Training**: 500,000 tokens  
**Evaluation Dataset**: wikitext (fallback from finance dataset)

| Metric | Value | SAEBench Standard | Status |
|--------|-------|-------------------|---------|
| **Loss Recovered** | 7.23% | ≥60-70% | ❌ Below threshold |
| **L0 Sparsity** | 227.37 | 20-200 (sweet spot: 40-120) | ❌ Outside range |
| **Dead Features** | 88.83% | ≤10-20% | ❌ Too many dead features |
| **Feature Absorption** | 1.000 | ≤0.25 (borderline: 0.25-0.35) | ❌ High absorption |

### Overall Assessment: 0/4 metrics healthy ❌

## 🔍 Analysis and Findings

### Why Results Differ from Training Metrics

1. **Different Evaluation Methodologies**
   - Training shows loss metrics on training data
   - Evaluation uses SAEBench methodology with real activations

2. **Insufficient Training**
   - 500k tokens is far too little for a 131k feature SAE
   - Expected: 10-50 million tokens minimum
   - Current: Only 500k tokens (10-100x less than needed)

3. **Training Configuration Issues**
   - No normalization (`normalize_activations: none`)
   - High L1 coefficient (5.0) may be too aggressive
   - Large expansion factor (32x) requires more training

4. **Feature Utilization Problems**
   - 88.8% dead features indicate poor capacity utilization
   - High L0 sparsity (227) shows over-activation of few features
   - High feature absorption (1.000) indicates redundant features

### Expected vs Actual Performance

| Aspect | Expected | Actual | Reason |
|--------|----------|--------|---------|
| Loss Recovery | 60-80% | 7.23% | Undertrained |
| Dead Features | <20% | 88.83% | Insufficient training |
| L0 Sparsity | 40-120 | 227.37 | Poor feature distribution |
| Training Time | 10-50M tokens | 500k tokens | Severely undertrained |

## 🛠️ Evaluation Script

### Usage

```bash
# Evaluate SAE with real model activations
python evaluate_sae_real_activations.py \
    --sae_path "checkpoints/llama_layer12_sae_saelens_500k_8d0fb52d/y05st1pp/final_500736" \
    --model_name "meta-llama/Meta-Llama-3-8B" \
    --layer 12 \
    --dataset "wikitext" \
    --num_samples 100 \
    --context_length 64 \
    --batch_size 16 \
    --verbose
```

### Key Features

- **Real Model Activations**: Uses actual model activations for evaluation
- **SAEBench Methodology**: Aligns with established SAE evaluation standards
- **GPU Optimization**: Efficient computation on CUDA devices
- **Comprehensive Metrics**: Loss recovery, sparsity, dead features, absorption
- **Memory Management**: Handles large SAEs with sampling strategies

## 🔧 Recommendations for Improvement

### 1. Increase Training Duration
```bash
# Recommended training parameters
--training_tokens 10000000  # 10M tokens minimum
--training_tokens 50000000  # 50M tokens for optimal performance
```

### 2. Adjust Hyperparameters
```bash
# Better configuration
--l1_coefficient 1.0        # Lower from 5.0
--normalize_activations expected_average_only_in
--lr 5e-5                   # Lower learning rate
```

### 3. Use Proper Normalization
```bash
# Try different normalization strategies
--normalize_activations expected_average_only_in
--normalize_activations expected_average_only_out
--normalize_activations expected_average_in_and_out
```

### 4. Optimize Architecture
```bash
# Consider smaller expansion factors
--expansion_factor 16       # Instead of 32
--expansion_factor 8        # For faster training
```

## 📚 Methodology

### SAEBench Evaluation Standards

1. **Loss Recovered (FVU)**: Fraction of variance unexplained
   - Healthy: ≥60-70%
   - Current: 7.23%

2. **L0 Sparsity**: Average number of active features per sample
   - Healthy: 20-200 (sweet spot: 40-120)
   - Current: 227.37

3. **Dead Features**: Percentage of features never activated
   - Healthy: ≤10-20%
   - Current: 88.83%

4. **Feature Absorption**: Cosine similarity between decoder weights
   - Healthy: ≤0.25 (borderline: 0.25-0.35)
   - Current: 1.000

### Real Activation Evaluation

- Uses actual model activations from target layer
- Processes activations through trained SAE
- Calculates reconstruction quality and feature utilization
- Provides comprehensive health assessment

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size and context length
   --batch_size 8 --context_length 32
   ```

2. **Dataset Loading Errors**
   ```bash
   # Use fallback datasets
   --dataset "wikitext"  # Instead of finance dataset
   ```

3. **SAE Configuration Issues**
   ```bash
   # Check checkpoint structure
   ls checkpoints/llama_layer12_sae_saelens_500k_8d0fb52d/y05st1pp/final_500736/
   ```

### Performance Optimization

1. **GPU Memory Management**
   - Use smaller batch sizes for large SAEs
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Training Efficiency**
   - Increase buffer size for better data utilization
   - Use proper normalization for stability
   - Monitor dead features during training

## 📈 Future Work

1. **Extended Training**: Train SAEs for longer durations (10-50M tokens)
2. **Hyperparameter Tuning**: Optimize L1 coefficient, learning rate, normalization
3. **Architecture Exploration**: Test different expansion factors and layer depths
4. **Multi-Layer Analysis**: Evaluate SAEs across multiple model layers
5. **Comparative Studies**: Compare SAELens with other SAE libraries (Sparsify, SAEBench)

## 🤝 Contributing

This project demonstrates SAE training and evaluation methodologies. Contributions are welcome for:
- Improved evaluation metrics
- Better training configurations
- Additional SAE libraries support
- Performance optimizations

## 📄 License

This project is for research and educational purposes. Please ensure compliance with model licenses (Meta-Llama-3-8B) and dataset terms of use.

---

**Note**: The current SAE is severely undertrained. Results are expected to improve significantly with proper training duration and hyperparameter tuning.
