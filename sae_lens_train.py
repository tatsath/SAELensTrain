#!/usr/bin/env python3
"""
Simple SAE Training Script
Run with: torchrun --nproc_per_node=N simple_train.py [options]
"""

import os
import torch
import torch.distributed as dist
import argparse
from pathlib import Path

# Import SAELens components
try:
    import sae_lens
    from sae_lens import LanguageModelSAERunnerConfig, StandardTrainingSAEConfig, LoggingConfig, LanguageModelSAETrainingRunner
    SAELENS_AVAILABLE = True
    print(f"✅ SAELens version: {getattr(sae_lens, '__version__', 'unknown')}")
except ImportError as e:
    print(f"❌ SAELens not available: {e}")
    print("Please install: pip install sae-lens")
    SAELENS_AVAILABLE = False
    exit(1)

def get_model_dimensions(model_name):
    """Get model dimensions automatically"""
    try:
        from transformer_lens import HookedTransformer
        model = HookedTransformer.from_pretrained(model_name, device="cpu")
        return {
            "d_mlp": model.cfg.d_mlp,
            "d_model": model.cfg.d_model,
            "n_layers": model.cfg.n_layers,
            "n_ctx": model.cfg.n_ctx
        }
    except Exception as e:
        print(f"⚠️  Could not auto-detect model dimensions: {e}")
        return None

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Simple SAE Training")
    
    # Model and Data
    parser.add_argument("--model_name", type=str, default="tiny-stories-1L-21M", help="Model name")
    parser.add_argument("--dataset_path", type=str, default="apollo-research/roneneldan-TinyStories-tokenizer-gpt2", help="Dataset path")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset config name (e.g., 'wikitext-2-raw-v1')")
    parser.add_argument("--is_dataset_tokenized", action="store_true", default=True, help="Dataset is tokenized")
    parser.add_argument("--dataset_trust_remote_code", action="store_true", default=False, help="Trust remote code for dataset loading")
    
    # Layer and Hook Configuration
    parser.add_argument("--layer", type=int, default=0, help="Layer number (0-indexed)")
    parser.add_argument("--hook_name", type=str, default=None, help="Hook name (auto-generated if not provided)")
    parser.add_argument("--auto_detect_dimensions", action="store_true", help="Auto-detect model dimensions")
    
    # SAE Parameters
    parser.add_argument("--d_in", type=int, default=None, help="Input dimension (auto-detected if not provided)")
    parser.add_argument("--d_sae", type=int, default=None, help="SAE hidden dimension")
    parser.add_argument("--expansion_factor", type=int, default=16, help="Expansion factor (d_sae = d_in * expansion_factor)")
    parser.add_argument("--l1_coefficient", type=float, default=5.0, help="L1 sparsity coefficient")
    parser.add_argument("--normalize_activations", type=str, default="expected_average_only_in", 
                       choices=["none", "expected_average_only_in", "constant_norm_rescale", "layer_norm"],
                       help="Activation normalization strategy")
    
    # Training Parameters
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--train_batch_size_tokens", type=int, default=4096, help="Training batch size in tokens")
    parser.add_argument("--context_size", type=int, default=None, help="Context size (auto-detected if not provided)")
    parser.add_argument("--n_batches_in_buffer", type=int, default=64, help="Number of batches in buffer")
    parser.add_argument("--training_tokens", type=int, default=1_000_000_000, help="Total training tokens (1B tokens)")
    parser.add_argument("--store_batch_size_prompts", type=int, default=16, help="Store batch size for prompts")
    
    # WandB
    parser.add_argument("--wandb_project", type=str, default="sae_lens_tutorial", help="WandB project name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    
    # GPU Configuration
    parser.add_argument("--cuda_devices", type=str, default=None, help="CUDA devices to use (e.g., '0,1,2,3')")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs for distributed training")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_checkpoints", type=int, default=0, help="Number of checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="Checkpoint path")
    parser.add_argument("--output_name", type=str, default=None, help="Custom output name for the SAE model (auto-generated if not provided)")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type")
    
    return parser.parse_args()

def setup_distributed(cuda_devices=None, num_gpus=1):
    """Setup distributed training if running under torchrun"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        # Initialize distributed training
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
        # Set device based on local rank
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        
        print(f"🚀 Distributed training: rank {rank}/{world_size}, device {device}")
        return device, rank, world_size
    else:
        # Single GPU training
        if cuda_devices:
            # Set CUDA devices if specified
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
            print(f"🔧 Set CUDA_VISIBLE_DEVICES={cuda_devices}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Single GPU training on {device}")
        return device, 0, 1

def main():
    """Main training function"""
    # Parse arguments first
    args = parse_args()
    
    # Setup distributed training
    device, rank, world_size = setup_distributed(args.cuda_devices, args.num_gpus)
    
    # Check SAELens availability after distributed setup
    if not SAELENS_AVAILABLE:
        if rank == 0:
            print("❌ SAELens not available. Exiting.")
        return
    
    # Parse arguments
    args = parse_args()
    
    # Auto-detect model dimensions if requested or if d_in is not provided
    model_info = None
    if args.auto_detect_dimensions or args.d_in is None or args.context_size is None:
        print(f"🔍 Auto-detecting dimensions for model: {args.model_name}")
        model_info = get_model_dimensions(args.model_name)
        
        if model_info:
            if args.d_in is None:
                args.d_in = model_info["d_model"]  # Use hidden size, not MLP size
                print(f"✅ Auto-detected d_in: {args.d_in}")
            
            if args.context_size is None:
                args.context_size = model_info["n_ctx"]
                print(f"✅ Auto-detected context_size: {args.context_size}")
    
    # Auto-generate hook name if not provided
    if args.hook_name is None:
        args.hook_name = f"blocks.{args.layer}.hook_resid_post"  # Use residual stream hook
        print(f"✅ Auto-generated hook_name: {args.hook_name}")
    
    # Calculate d_sae if not provided
    if args.d_sae is None:
        args.d_sae = args.d_in * args.expansion_factor
        print(f"✅ Calculated d_sae: {args.d_sae} ({args.d_in} * {args.expansion_factor})")
    
    # Generate custom checkpoint path if output_name is provided
    if args.output_name:
        import uuid
        # Create a unique identifier for this training run
        run_id = str(uuid.uuid4())[:8]
        custom_checkpoint_path = f"{args.checkpoint_path}/{args.output_name}_{run_id}"
        print(f"✅ Custom output path: {custom_checkpoint_path}")
    else:
        custom_checkpoint_path = args.checkpoint_path
    
    # Ensure checkpoint directory exists
    os.makedirs(custom_checkpoint_path, exist_ok=True)
    print(f"✅ Created checkpoint directory: {custom_checkpoint_path}")
    
    # Create configuration with new v6 nested structure
    cfg = LanguageModelSAERunnerConfig(
        # SAE Parameters (nested)
        sae=StandardTrainingSAEConfig(
            d_in=args.d_in,
            d_sae=args.d_sae,
            normalize_activations=args.normalize_activations,
            l1_coefficient=args.l1_coefficient,
            apply_b_dec_to_input=True,
        ),
        
        # Data Generating Function (Model + Training Distribution)
        model_name=args.model_name,
        hook_name=args.hook_name,
        dataset_path=args.dataset_path,
        dataset_trust_remote_code=args.dataset_trust_remote_code,
        streaming=True,
        is_dataset_tokenized=args.is_dataset_tokenized,
        context_size=args.context_size,
        prepend_bos=True,
        
        # Training Parameters
        lr=args.lr,
        train_batch_size_tokens=args.train_batch_size_tokens,
        store_batch_size_prompts=args.store_batch_size_prompts,
        training_tokens=args.training_tokens,
        n_batches_in_buffer=args.n_batches_in_buffer,
        
        # Learning Rate and Optimization
        lr_scheduler_name="constant",
        lr_warm_up_steps=args.training_tokens // (args.train_batch_size_tokens * 20),
        adam_beta1=0.9,
        adam_beta2=0.999,
        
        # Feature Management
        feature_sampling_window=2500,
        dead_feature_window=5000,
        dead_feature_threshold=1e-8,
        
        # Device and Precision
        device=device,
        act_store_device="cpu",
        dtype=args.dtype,
        autocast=True,
        
        # Compilation (for performance)
        compile_llm=False,
        compile_sae=False,
        sae_compilation_mode="max-autotune",
        
        # Logging (nested)
        logger=LoggingConfig(
            log_to_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
        ),
        
        # Checkpointing
        n_checkpoints=args.n_checkpoints,
        checkpoint_path=custom_checkpoint_path,
        
        # Reproducibility
        seed=args.seed,
        
        # Model Loading
        model_from_pretrained_kwargs={
            "trust_remote_code": True,
        }
    )
    
    # Print configuration summary
    if rank == 0:
        print("📋 Configuration Summary:")
        print(f"   Model: {cfg.model_name}")
        print(f"   Layer: {args.layer}")
        print(f"   Hook: {cfg.hook_name}")
        print(f"   Dataset: {cfg.dataset_path}")
        print(f"   SAE size: {cfg.sae.d_in} -> {cfg.sae.d_sae}")
        print(f"   Expansion factor: {args.expansion_factor}")
        print(f"   Training tokens: {cfg.training_tokens:,}")
        print(f"   Learning rate: {cfg.lr}")
        print(f"   L1 coefficient: {cfg.sae.l1_coefficient}")
        print(f"   Normalize activations: {cfg.sae.normalize_activations}")
        print(f"   Batch size: {cfg.train_batch_size_tokens}")
        print(f"   Context size: {cfg.context_size}")
        print(f"   WandB project: {cfg.logger.wandb_project}")
        print(f"   WandB logging: {cfg.logger.log_to_wandb}")
        print(f"   Checkpoint path: {cfg.checkpoint_path}")
        print(f"   Device: {device}")
        print(f"   World size: {world_size}")
        print()
    
    # Create training runner and start training
    try:
        runner = LanguageModelSAETrainingRunner(cfg=cfg)
        sae = runner.run()
        
        if rank == 0:
            print("✅ Training completed successfully!")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
    
    finally:
        # Clean up distributed training
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
