"""Training script for Legal AI Assistant."""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ModelConfig, ModelType
from config.training_config import TrainingConfig, DataConfig
from training.trainer import LegalAITrainer
from utils.logging_utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Legal AI Assistant"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-type",
        type=str,
        default="mistral",
        choices=["falcon", "llama2", "mistral", "gemma", "phi3"],
        help="Model type to use",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Specific model name (overrides model-type)",
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="nisaar/Lawyer_GPT_India",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to local dataset file",
    )
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="jsonl",
        choices=["jsonl", "json", "csv"],
        help="Dataset file format",
    )
    
    # Training arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for model",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )

    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map for model loading (e.g. 'auto', 'cpu', 'cuda:0'). Useful on Windows to force CPU or a specific GPU.",
    )

    # Runtime toggles
    parser.add_argument(
        "--disable-quantization",
        action="store_true",
        help="Disable bitsandbytes quantization (4/8-bit) for smoke runs or incompatible GPUs",
    )
    parser.add_argument(
        "--force-quantization",
        action="store_true",
        help="Force-enable bitsandbytes quantization even on Windows (use with caution)",
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    
    # Hub arguments
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="Model ID on HuggingFace Hub",
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--logging-dir",
        type=str,
        default="./logs",
        help="Logging directory",
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(
        log_file=os.path.join(args.logging_dir, "training.log")
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Legal AI Assistant Training")
    logger.info("=" * 80)
    
    # Map model type string to enum
    model_type_map = {
        "falcon": ModelType.FALCON_7B,
        "llama2": ModelType.LLAMA2_7B,
        "mistral": ModelType.MISTRAL_7B,
        "gemma": ModelType.GEMMA_7B,
        "phi3": ModelType.PHI3_MINI,
    }
    
    # Create configurations
    model_config = ModelConfig(
        model_type=model_type_map[args.model_type],
        model_name=args.model_name,
    )

    # Apply device_map override if provided
    if args.device_map:
        model_config.device_map = args.device_map

    # If user disabled quantization at runtime, ensure quant flags are off
    if args.disable_quantization:
        model_config.quantization.load_in_4bit = False
        model_config.quantization.load_in_8bit = False

    # On Windows, default to disabling quantization and Flash Attention because
    # bitsandbytes and flash_attn are often not supported or unstable on native
    # Windows installs. Allow override with --force-quantization.
    try:
        import platform

        if platform.system().lower().startswith("win") and not args.force_quantization:
            logger = logging.getLogger(__name__)
            logger.warning(
                "Detected Windows OS: disabling 4/8-bit quantization and Flash Attention by default. "
                "Use --force-quantization to override at your own risk."
            )
            model_config.quantization.load_in_4bit = False
            model_config.quantization.load_in_8bit = False
            model_config.use_flash_attention = False
    except Exception:
        # If platform detection fails for any reason, don't crash; keep defaults
        pass
    model_config.lora.r = args.lora_r
    model_config.lora.lora_alpha = args.lora_alpha
    model_config.lora.lora_dropout = args.lora_dropout
    
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )
    
    data_config = DataConfig(
        dataset_name=args.dataset_name,
        custom_dataset_path=args.dataset_path,
        dataset_format=args.dataset_format,
        max_length=args.max_seq_length,
        seed=args.seed,
    )

    # If the user passed a dataset path that looks like a Hugging Face hub id
    # (owner/repo) and it does not look like a local filesystem path, treat it
    # as a hub dataset name so the trainer will load from the Hub.
    if args.dataset_path:
        dp = args.dataset_path
        looks_like_windows_path = (os.name == "nt") and (
            (len(dp) >= 2 and dp[1] == ':') or ('\\' in dp)
        )
        looks_like_unix_abs = dp.startswith('/') or dp.startswith('./') or dp.startswith('../')

        if ('/' in dp) and not (looks_like_windows_path or looks_like_unix_abs):
            logger = logging.getLogger(__name__)
            logger.info(f"Interpreting dataset-path '{dp}' as a HuggingFace Hub dataset id.")
            data_config.dataset_name = dp
            data_config.custom_dataset_path = None
    
    # Log configurations
    logger.info(f"Model: {model_config.model_name}")
    logger.info(f"Dataset: {args.dataset_name or args.dataset_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    
    # Create trainer
    trainer = LegalAITrainer(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
    )

    # If a custom dataset path was provided, validate it when it looks like a
    # local filesystem path. We allow Hub dataset ids like 'owner/repo' here and
    # won't treat them as missing files.
    if args.dataset_path:
        dp = args.dataset_path
        looks_like_windows_path = False
        try:
            # Windows absolute path or drive-letter path like C:\...
            looks_like_windows_path = (os.name == "nt") and (
                (len(dp) >= 2 and dp[1] == ':') or ('\\' in dp)
            )
        except Exception:
            looks_like_windows_path = False

        looks_like_unix_abs = dp.startswith('/') or dp.startswith('./') or dp.startswith('../')

        # If it looks like a local path (absolute, drive-letter, or contains backslashes)
        # then require the file to exist. Otherwise assume it's a Hub dataset id and allow.
        if looks_like_windows_path or looks_like_unix_abs:
            if not os.path.exists(dp):
                logger.error(f"Dataset file not found: {dp}")
                raise FileNotFoundError(f"Dataset file not found: {dp}")
    
    # Run training
    try:
        trainer.run()
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()