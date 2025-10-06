"""Training configuration for Legal AI Assistant."""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Output and logging
    output_dir: str = "./output"
    logging_dir: str = "./logs"
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    
    # Training parameters
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 means use num_train_epochs
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    
    # Optimization
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    
    # Precision
    fp16: bool = False
    bf16: bool = True
    
    # Evaluation
    # Note: some transformers versions use different arg names for evaluation
    # strategy. To ensure compatibility across versions, we avoid passing the
    # evaluation keys and default to not loading best model at end.
    eval_steps: int = 100
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    
    # Gradient and memory optimization
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    
    # Data
    max_seq_length: int = 2048
    dataloader_num_workers: int = 4
    
    # Reproducibility
    seed: int = 42
    
    # Reporting
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    
    # Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None
    
    def to_training_arguments(self):
        """Convert to HuggingFace TrainingArguments compatible dict."""
        # Use a conservative whitelist of args supported across transformers
        # versions to avoid runtime TypeErrors when the library differs.
        args = {
            "output_dir": self.output_dir,
            "logging_dir": self.logging_dir,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "eval_steps": self.eval_steps,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "gradient_checkpointing": self.gradient_checkpointing,
            "max_grad_norm": self.max_grad_norm,
            "dataloader_num_workers": self.dataloader_num_workers,
            "seed": self.seed,
            "report_to": self.report_to,
            "push_to_hub": self.push_to_hub,
            "hub_model_id": self.hub_model_id,
            "hub_token": self.hub_token,
        }

        # Some transformers versions accept additional optional keys; keep
        # backward compatibility by adding them only when present on the
        # TrainingArguments constructor. We won't attempt reflection here; the
        # conservative set above is usually sufficient.
        return args


@dataclass
class DataConfig:
    """Configuration for dataset."""
    dataset_name: str = "nisaar/Lawyer_GPT_India"
    dataset_split: str = "train"
    train_split_ratio: float = 0.9
    validation_split_ratio: float = 0.1
    test_split_ratio: float = 0.0
    shuffle_dataset: bool = True
    seed: int = 42
    
    # Preprocessing
    max_length: int = 2048
    padding: str = "max_length"
    truncation: bool = True
    
    # Custom dataset path
    custom_dataset_path: Optional[str] = None
    dataset_format: str = "jsonl"  # jsonl, json, csv