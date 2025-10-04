import logging
from typing import Dict, List, Optional
from datasets import Dataset, DatasetDict
from data.dataset_loader import DatasetLoader
from transformers import PreTrainedTokenizer

from data.prompt_templates import get_prompt_template

logger = logging.getLogger(__name__)


class DataProcessor:
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        prompt_template: str = "legal_assistant",
    ):
        """DataProcessor for loading, processing, and tokenizing datasets."""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = get_prompt_template(prompt_template)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_dataset_from_hub(
        self,
        dataset_name: str,
        split: str = "train",
    ) -> Dataset:
        """Load dataset from Hugging Face Hub."""
        logger.info(f"Loading dataset {dataset_name} (split: {split}) via DatasetLoader")
        return DatasetLoader.load_from_hub(dataset_name, split=split)
    
    def load_dataset_from_file(
        self,
        file_path: str,
        file_format: Optional[str] = None,
    ) -> Dataset:
        """Load dataset from local file using DatasetLoader."""
        logger.info(f"Loading dataset from {file_path} via DatasetLoader")
        return DatasetLoader.load_from_file(file_path, file_format)
    
    def split_dataset(
        self,
        dataset: Dataset,
        train_ratio: float = 0.9,
        val_ratio: float = 0.1,
        test_ratio: float = 0.0,
        seed: int = 42,
    ) -> DatasetDict:
        """Split dataset into train, validation, and test sets."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        
        logger.info(f"Splitting dataset (train: {train_ratio}, val: {val_ratio}, test: {test_ratio})")
        
        # First split into train and temp
        train_test = dataset.train_test_split(
            test_size=(1 - train_ratio),
            seed=seed,
        )
        
        splits = {"train": train_test["train"]}
        
        if test_ratio > 0:
            # Split temp into validation and test
            val_test_ratio = val_ratio / (val_ratio + test_ratio)
            val_test = train_test["test"].train_test_split(
                test_size=(1 - val_test_ratio),
                seed=seed,
            )
            splits["validation"] = val_test["train"]
            splits["test"] = val_test["test"]
        else:
            splits["validation"] = train_test["test"]
        
        dataset_dict = DatasetDict(splits)
        
        for split_name, split_data in dataset_dict.items():
            logger.info(f"{split_name}: {len(split_data)} examples")
        
        return dataset_dict
    
    def format_prompt(self, example: Dict) -> str:
        """Format example using the prompt template."""
        return self.prompt_template.format(example)
    
    def tokenize_function(self, example: Dict) -> Dict:
        """Tokenize a single example after formatting the prompt."""
        full_prompt = self.format_prompt(example)
        
        tokenized = self.tokenizer(
            full_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,
        )
        
        # Add labels (same as input_ids for causal LM)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def prepare_dataset(
        self,
        dataset: Dataset,
        num_proc: int = 4,
        remove_columns: Optional[List[str]] = None,
    ) -> Dataset:
        """Tokenize and prepare dataset for training."""
        logger.info("Tokenizing dataset...")
        
        if remove_columns is None:
            remove_columns = dataset.column_names
        
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            remove_columns=remove_columns,
            num_proc=num_proc,
            desc="Tokenizing",
        )
        
        logger.info(f"Tokenization complete. Dataset size: {len(tokenized_dataset)}")
        return tokenized_dataset
    
    def validate_dataset(self, dataset: Dataset) -> bool:
        """Validate that dataset has required fields for the prompt template."""
        required_fields = self.prompt_template.get_required_fields()
        
        if not dataset:
            raise ValueError("Dataset is empty")
        
        # Check first example has required fields
        first_example = dataset[0]
        missing_fields = [f for f in required_fields if f not in first_example]
        
        if missing_fields:
            raise ValueError(
                f"Dataset missing required fields: {missing_fields}. "
                f"Required fields: {required_fields}"
            )
        
        logger.info("Dataset validation passed")
        return True