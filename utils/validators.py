"""Validation utilities for Legal AI Assistant."""
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validate configuration objects."""
    
    @staticmethod
    def validate_model_config(config: Any) -> bool:
       
        # Check required attributes
        required_attrs = ['model_name', 'device_map']
        for attr in required_attrs:
            if not hasattr(config, attr):
                raise ValueError(f"Model config missing required attribute: {attr}")
        
        # Validate LoRA config
        if hasattr(config, 'lora'):
            lora = config.lora
            if lora.r <= 0:
                raise ValueError(f"LoRA rank must be positive, got {lora.r}")
            if lora.lora_alpha <= 0:
                raise ValueError(f"LoRA alpha must be positive, got {lora.lora_alpha}")
            if not (0 <= lora.lora_dropout < 1):
                raise ValueError(f"LoRA dropout must be in [0, 1), got {lora.lora_dropout}")
        
        logger.info("Model configuration validated successfully")
        return True
    
    @staticmethod
    def validate_training_config(config: Any) -> bool:
        
        # Validate epochs/steps
        if config.num_train_epochs <= 0 and config.max_steps <= 0:
            raise ValueError(
                "Either num_train_epochs or max_steps must be positive"
            )
        
        # Validate batch sizes
        if config.per_device_train_batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        # Validate learning rate
        if config.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        # Validate warmup ratio
        if not (0 <= config.warmup_ratio < 1):
            raise ValueError(f"Warmup ratio must be in [0, 1), got {config.warmup_ratio}")
        
        logger.info("Training configuration validated successfully")
        return True
    
    @staticmethod
    def validate_data_config(config: Any) -> bool:
       
        # Validate split ratios
        total_split = (
            config.train_split_ratio +
            config.validation_split_ratio +
            config.test_split_ratio
        )
        
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total_split}"
            )
        
        # Validate max length
        if config.max_length <= 0:
            raise ValueError("Max length must be positive")
        
        # Check dataset source
        if not config.dataset_name and not config.custom_dataset_path:
            raise ValueError(
                "Either dataset_name or custom_dataset_path must be provided"
            )
        
        logger.info("Data configuration validated successfully")
        return True


class DatasetValidator:
    """Validate dataset structure and content."""
    
    @staticmethod
    def validate_example(
        example: Dict[str, Any],
        required_fields: List[str],
    ) -> bool:
        
        # Check required fields
        missing_fields = [
            field for field in required_fields
            if field not in example
        ]
        
        if missing_fields:
            raise ValueError(
                f"Example missing required fields: {missing_fields}"
            )
        
        # Check field types and content
        for field in required_fields:
            value = example[field]
            
            if not isinstance(value, str):
                raise ValueError(
                    f"Field '{field}' must be string, got {type(value)}"
                )
            
            if not value.strip():
                raise ValueError(
                    f"Field '{field}' cannot be empty"
                )
        
        return True
    
    @staticmethod
    def validate_dataset(
        dataset: Any,
        required_fields: List[str],
        min_examples: int = 1,
    ) -> bool:
       
        # Check dataset size
        if len(dataset) < min_examples:
            raise ValueError(
                f"Dataset must have at least {min_examples} examples, "
                f"got {len(dataset)}"
            )
        
        # Validate first few examples
        num_to_check = min(10, len(dataset))
        
        for i in range(num_to_check):
            try:
                DatasetValidator.validate_example(
                    dataset[i],
                    required_fields,
                )
            except ValueError as e:
                raise ValueError(f"Invalid example at index {i}: {e}")
        
        logger.info(
            f"Dataset validated successfully: {len(dataset)} examples"
        )
        return True


class PathValidator:
    """Validate file and directory paths."""
    
    @staticmethod
    def validate_file_exists(file_path: str) -> bool:
        
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        return True
    
    @staticmethod
    def validate_dir_exists(dir_path: str, create: bool = False) -> bool:
      
        path = Path(dir_path)
        
        if not path.exists():
            if create:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            else:
                raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {dir_path}")
        
        return True
    
    @staticmethod
    def validate_output_path(
        file_path: str,
        create_parent: bool = True,
    ) -> bool:
       
        path = Path(file_path)
        
        # Create parent directory if needed
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if parent directory exists
        if not path.parent.exists():
            raise FileNotFoundError(
                f"Parent directory not found: {path.parent}"
            )
        
        return True


class SystemValidator:
    """Validate system requirements."""
    
    @staticmethod
    def validate_cuda_available() -> bool:
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Training will be slow on CPU.")
            return False
        
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        return True
    
    @staticmethod
    def validate_gpu_memory(min_memory_gb: float = 8.0) -> bool:
       
        if not torch.cuda.is_available():
            return False
        
        total_memory = torch.cuda.get_device_properties(0).total_memory
        total_memory_gb = total_memory / (1024 ** 3)
        
        if total_memory_gb < min_memory_gb:
            logger.warning(
                f"GPU memory ({total_memory_gb:.1f} GB) is less than "
                f"recommended minimum ({min_memory_gb:.1f} GB)"
            )
            return False
        
        logger.info(f"GPU memory available: {total_memory_gb:.1f} GB")
        return True
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        
        info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "pytorch_version": torch.__version__,
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory
            info["gpu_memory_gb"] = total_memory / (1024 ** 3)
        
        return info