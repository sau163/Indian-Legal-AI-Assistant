"""Training module for Legal AI Assistant."""
import logging
import os
from typing import Optional
 
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import DatasetDict

from config.model_config import ModelConfig
from config.training_config import TrainingConfig, DataConfig
from models.model_factory import ModelFactory
from data.data_processor import DataProcessor

logger = logging.getLogger(__name__)


class LegalAITrainer:
    """Trainer for Legal AI Assistant."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig,
    ):
        """Initialize trainer.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            data_config: Data configuration
        """
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None
    
    def setup(self) -> None:
        """Setup model, tokenizer, and data."""
        logger.info("Setting up trainer...")
        
        # Load model and tokenizer
        self.model, self.tokenizer = ModelFactory.load_base_model(
            self.model_config
        )
        
        # Prepare model for training
        self.model = ModelFactory.prepare_model_for_training(
            self.model,
            self.model_config,
        )
        
        # Load and prepare dataset
        self._prepare_dataset()
        
        # Create trainer
        self._create_trainer()
        
        logger.info("Setup complete")
    
    def _prepare_dataset(self) -> None:
        """Load and prepare dataset."""
        logger.info("Preparing dataset...")
        
        # Initialize data processor
        processor = DataProcessor(
            tokenizer=self.tokenizer,
            max_length=self.data_config.max_length,
            prompt_template="legal_assistant",
        )
        
        # Load dataset
        if self.data_config.custom_dataset_path:
            dataset = processor.load_dataset_from_file(
                self.data_config.custom_dataset_path,
                self.data_config.dataset_format,
            )
        else:
            dataset = processor.load_dataset_from_hub(
                self.data_config.dataset_name,
                self.data_config.dataset_split,
            )
        
        # Validate dataset
        processor.validate_dataset(dataset)
        
        # Shuffle if requested
        if self.data_config.shuffle_dataset:
            dataset = dataset.shuffle(seed=self.data_config.seed)
        
        # Split dataset
        self.dataset = processor.split_dataset(
            dataset,
            train_ratio=self.data_config.train_split_ratio,
            val_ratio=self.data_config.validation_split_ratio,
            test_ratio=self.data_config.test_split_ratio,
            seed=self.data_config.seed,
        )
        
        # Tokenize dataset
        self.dataset = DatasetDict({
            split: processor.prepare_dataset(data, num_proc=4)
            for split, data in self.dataset.items()
        })
        
        logger.info("Dataset preparation complete")
    
    def _create_trainer(self) -> None:
        """Create HuggingFace Trainer."""
        logger.info("Creating trainer...")
        
        # Create output directory
        os.makedirs(self.training_config.output_dir, exist_ok=True)
        
        # Training arguments
        training_kwargs = self.training_config.to_training_arguments()

        # If the loaded model is on CPU, force Trainer to not use CUDA even
        # when a GPU is available on the system. This prevents unexpected
        # CUDA kernel calls when we're intentionally running on CPU.
        try:
            model_device = getattr(self.model, "device", None)
            if model_device is not None and str(model_device).startswith("cpu"):
                training_kwargs["no_cuda"] = True
                logger.info("Model on CPU: forcing Trainer to run with no_cuda=True")
        except Exception:
            # If detection fails, don't crash; rely on defaults
            pass

        training_args = TrainingArguments(**training_kwargs)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset.get("validation"),
            data_collator=data_collator,
        )
        
        logger.info("Trainer created")
    
    def train(self) -> None:
        """Train the model."""
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")
        
        logger.info("Starting training...")
        
        # Disable caching for training
        self.model.config.use_cache = False
        
        # Train
        train_result = self.trainer.train()
        
        # Log metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        logger.info("Training complete")
        logger.info(f"Training metrics: {metrics}")
    
    def evaluate(self) -> dict:
        """Evaluate the model.
        
        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")
        
        logger.info("Evaluating model...")
        
        metrics = self.trainer.evaluate()
        
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def save_model(
        self,
        output_dir: Optional[str] = None,
    ) -> None:
        """Save the trained model.
        
        Args:
            output_dir: Output directory (uses training config if None)
        """
        if output_dir is None:
            output_dir = self.training_config.output_dir
        
        logger.info(f"Saving model to {output_dir}")
        
        # Save model
        ModelFactory.save_model(
            model=self.model,
            tokenizer=self.tokenizer,
            output_dir=output_dir,
            push_to_hub=self.training_config.push_to_hub,
            hub_model_id=self.training_config.hub_model_id,
        )
        
        # Save trainer state
        self.trainer.save_state()
        
        logger.info("Model saved successfully")
    
    def run(self) -> None:
        """Run full training pipeline."""
        logger.info("=" * 80)
        logger.info("Starting Legal AI Training Pipeline")
        logger.info("=" * 80)
        
        # Setup
        self.setup()
        
        # Train
        self.train()
        
        # Evaluate
        if "validation" in self.dataset:
            self.evaluate()
        
        # Save
        self.save_model()
        
        logger.info("=" * 80)
        logger.info("Training pipeline complete!")
        logger.info("=" * 80)