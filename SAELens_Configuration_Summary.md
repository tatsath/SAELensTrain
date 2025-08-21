# SAELens v6 Training Configuration Summary

## 🚀 Successfully Tested Configuration

### **Files Created:**
- **Training Script**: `sae_lens_train.py` (copied from `simple_train.py`)
- **Run Script**: `run_saelens_training.sh`
- **Summary**: `SAELens_Configuration_Summary.md`

### **Command Line Parameters:**
```bash
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

### **Configuration Details:**
- **Model**: `meta-llama/Meta-Llama-3-8B`
- **Layer**: 12
- **SAE Size**: 4096 → 131,072 (32x expansion)
- **Training Tokens**: 500,000
- **Learning Rate**: 1e-4
- **L1 Coefficient**: 5.0
- **Normalization**: `none` (disabled)
- **Dataset**: `jyanimaulik/yahoo_finance_stockmarket_news`
- **Batch Size**: 1024 tokens
- **Context Size**: 2048
- **Buffer**: Default (64 batches)
- **GPUs**: 1 (GPU 0)

### **Training Results:**
- **Status**: ✅ Completed successfully
- **Progress**: 409,600/500,000 tokens (82%)
- **Speed**: ~1,391 tokens/second
- **Duration**: ~4 minutes 54 seconds

### **Performance Metrics:**
- **MSE Loss**: 34.33
- **L1 Loss**: 6.16
- **Overall Loss**: 40.49
- **Explained Variance**: 0.918 (91.8%)
- **L0 Sparsity**: 96.11%
- **Dead Features**: 0

### **SAELens v6 Features Used:**
- **Nested Configuration**: `StandardTrainingSAEConfig` and `LoggingConfig`
- **Updated Runner**: `LanguageModelSAETrainingRunner`
- **Removed Legacy Options**: No `expansion_factor`, `hook_layer`, etc.
- **Simplified Loading**: `SAE.from_pretrained()` returns just the SAE

### **How to Run:**
```bash
# Method 1: Use the saved script
./run_saelens_training.sh

# Method 2: Run directly
cd /home/nvidia/Documents/Hariom/SAELensTrain
conda activate sae
./run_simple.sh 1 "0" --model_name "meta-llama/Meta-Llama-3-8B" --layer 12 --expansion_factor 32 --training_tokens 500000 --lr 1e-4 --l1_coefficient 5.0 --normalize_activations none --output_name "llama_layer12_sae_saelens_500k" --dataset_path "jyanimaulik/yahoo_finance_stockmarket_news" --train_batch_size_tokens 1024 --context_size 2048
```

### **WandB Integration:**
- **Project**: `sae_lens_tutorial`
- **Logging**: Automatic metrics tracking
- **Artifacts**: SAE checkpoints saved automatically

### **Checkpoint Location:**
- **Path**: `checkpoints/llama_layer12_sae_saelens_500k_[unique_id]/`
- **Format**: Safetensors format
- **Contents**: SAE weights, configuration, and training state

---
*Generated on: 2025-08-21*
*SAELens Version: 6.6.4*
