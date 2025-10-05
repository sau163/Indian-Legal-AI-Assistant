"""Custom callbacks for training Legal AI Assistant."""
import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

logger = logging.getLogger(__name__)


class MetricsLoggingCallback(TrainerCallback):
    """Callback for enhanced metrics logging."""
    
    def __init__(self, log_interval: int = 10):
        """Initialize callback.
        
        Args:
            log_interval: Number of steps between detailed logs
        """
        self.log_interval = log_interval
        self.start_time = None
        self.step_times = []
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the beginning of training."""
        self.start_time = time.time()
        logger.info("=" * 80)
        logger.info("Training Started")
        logger.info("=" * 80)
        logger.info(f"Total steps: {state.max_steps}")
        logger.info(f"Batch size: {args.per_device_train_batch_size}")
        logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info("=" * 80)
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the end of each training step."""
        if state.global_step % self.log_interval == 0:
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            # Calculate progress
            progress = (state.global_step / state.max_steps) * 100
            
            # Estimate remaining time
            steps_per_second = state.global_step / elapsed if elapsed > 0 else 0
            remaining_steps = state.max_steps - state.global_step
            eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0
            
            logger.info(
                f"Step {state.global_step}/{state.max_steps} ({progress:.1f}%) | "
                f"Loss: {state.log_history[-1].get('loss', 0.0):.4f} | "
                f"Speed: {steps_per_second:.2f} steps/s | "
                f"ETA: {eta_seconds/60:.1f} min"
            )
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the end of training."""
        total_time = time.time() - self.start_time
        logger.info("=" * 80)
        logger.info("Training Completed")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_time/60:.2f} minutes")
        logger.info(f"Total steps: {state.global_step}")
        logger.info(f"Average time per step: {total_time/state.global_step:.2f} seconds")
        logger.info("=" * 80)


class EarlyStoppingCallback(TrainerCallback):
    """Callback for early stopping based on evaluation metrics."""
    
    def __init__(
        self,
        patience: int = 3,
        threshold: float = 0.0,
        metric_name: str = "eval_loss",
        mode: str = "min",
    ):
        """Initialize callback.
        
        Args:
            patience: Number of evaluations to wait before stopping
            threshold: Minimum improvement threshold
            metric_name: Name of metric to monitor
            mode: 'min' for metrics that should decrease, 'max' for increase
        """
        self.patience = patience
        self.threshold = threshold
        self.metric_name = metric_name
        self.mode = mode
        
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.wait_count = 0
    
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs,
    ):
        """Called after evaluation."""
        if self.metric_name not in metrics:
            return
        
        current_metric = metrics[self.metric_name]
        
        # Check if improved
        if self.mode == 'min':
            improved = current_metric < (self.best_metric - self.threshold)
        else:
            improved = current_metric > (self.best_metric + self.threshold)
        
        if improved:
            self.best_metric = current_metric
            self.wait_count = 0
            logger.info(
                f"✓ New best {self.metric_name}: {current_metric:.4f}"
            )
        else:
            self.wait_count += 1
            logger.info(
                f"No improvement in {self.metric_name}. "
                f"Patience: {self.wait_count}/{self.patience}"
            )
            
            if self.wait_count >= self.patience:
                logger.info("Early stopping triggered!")
                control.should_training_stop = True


class CheckpointCleanupCallback(TrainerCallback):
    """Callback to clean up old checkpoints."""
    
    def __init__(self, keep_last_n: int = 3):
        """Initialize callback.
        
        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        self.keep_last_n = keep_last_n
    
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called after saving a checkpoint."""
        output_dir = Path(args.output_dir)
        
        # Get all checkpoint directories
        checkpoints = sorted(
            [d for d in output_dir.glob("checkpoint-*") if d.is_dir()],
            key=lambda x: int(x.name.split("-")[1]),
        )
        
        # Remove old checkpoints
        if len(checkpoints) > self.keep_last_n:
            for checkpoint in checkpoints[:-self.keep_last_n]:
                logger.info(f"Removing old checkpoint: {checkpoint}")
                import shutil
                shutil.rmtree(checkpoint)


class GPUMemoryCallback(TrainerCallback):
    """Callback to monitor GPU memory usage."""
    
    def __init__(self, log_interval: int = 100):
        """Initialize callback.
        
        Args:
            log_interval: Steps between memory logs
        """
        self.log_interval = log_interval
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the end of each step."""
        import torch
        
        if not torch.cuda.is_available():
            return
        
        if state.global_step % self.log_interval == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            
            logger.info(
                f"GPU Memory - Allocated: {allocated:.2f} GB, "
                f"Reserved: {reserved:.2f} GB"
            )


class CustomLoggingCallback(TrainerCallback):
    """Callback for custom logging to file."""
    
    def __init__(self, log_file: str):
        """Initialize callback.
        
        Args:
            log_file: Path to log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Clear existing log file
        with open(self.log_file, 'w') as f:
            f.write("step,loss,learning_rate,epoch\n")
    
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float],
        **kwargs,
    ):
        """Called when logging."""
        if "loss" in logs:
            with open(self.log_file, 'a') as f:
                f.write(
                    f"{state.global_step},"
                    f"{logs.get('loss', 0.0)},"
                    f"{logs.get('learning_rate', 0.0)},"
                    f"{state.epoch}\n"
                )


def get_default_callbacks(
    output_dir: str,
    patience: int = 3,
    keep_checkpoints: int = 3,
) -> list:
    """Get default callbacks for training.
    
    Args:
        output_dir: Output directory for logs
        patience: Early stopping patience
        keep_checkpoints: Number of checkpoints to keep
        
    Returns:
        List of callback instances
    """
    return [
        MetricsLoggingCallback(log_interval=10),
        EarlyStoppingCallback(patience=patience),
        CheckpointCleanupCallback(keep_last_n=keep_checkpoints),
        GPUMemoryCallback(log_interval=100),
        CustomLoggingCallback(log_file=f"{output_dir}/training_log.csv"),
    ]