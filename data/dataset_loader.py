"""Dataset loader for Legal AI Assistant."""
import json
import logging
from pathlib import Path
from typing import Optional
from datasets import Dataset, load_dataset
import pandas as pd

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load datasets from various sources."""
    
    @staticmethod
    def load_from_hub(
        dataset_name: str,
        split: str = "train",
        cache_dir: Optional[str] = None,
    ) -> Dataset:
       
        logger.info(f"Loading dataset from Hub: {dataset_name}")
        
        try:
            dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=cache_dir,
            )
            logger.info(f"Successfully loaded {len(dataset)} examples")
            return dataset
        
        except Exception as e:
            logger.error(
                f"Failed to load dataset from Hub: {e}.\n"
                f"Make sure `dataset_name` is a HF dataset id like 'owner/dataset_name' or a valid local dataset path."
            )
            raise
    
    @staticmethod
    def load_from_jsonl(file_path: str) -> Dataset:
      
        logger.info(f"Loading dataset from JSONL: {file_path}")
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Remove trailing commas if present
                line = line.rstrip(',')
                
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Skipping invalid JSON on line {line_num}: {e}"
                    )
                    continue
        
        logger.info(f"Loaded {len(data)} examples from JSONL")
        return Dataset.from_list(data)
    
    @staticmethod
    def load_from_json(file_path: str) -> Dataset:
        
        logger.info(f"Loading dataset from JSON: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict):
            if "data" in data:
                data = data["data"]
            elif "questions" in data:
                data = data["questions"]
            else:
                # Assume it's a single example
                data = [data]
        
        if not isinstance(data, list):
            raise ValueError(
                f"Expected list of examples, got {type(data)}"
            )
        
        logger.info(f"Loaded {len(data)} examples from JSON")
        return Dataset.from_list(data)
    
    @staticmethod
    def load_from_csv(file_path: str) -> Dataset:
      
        logger.info(f"Loading dataset from CSV: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        # Reset index to avoid carrying the pandas index into the Dataset
        df = df.reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} examples from CSV")
        return Dataset.from_pandas(df)
    
    @staticmethod
    def load_from_file(
        file_path: str,
        file_format: Optional[str] = None,
    ) -> Dataset:
       
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Auto-detect format from extension
        if file_format is None:
            file_format = path.suffix.lstrip('.').lower()
        
        loaders = {
            'jsonl': DatasetLoader.load_from_jsonl,
            'json': DatasetLoader.load_from_json,
            'csv': DatasetLoader.load_from_csv,
        }
        
        if file_format not in loaders:
            raise ValueError(
                f"Unsupported file format: {file_format}. "
                f"Supported formats: {list(loaders.keys())}"
            )
        
        return loaders[file_format](file_path)
    
    @staticmethod
    def save_to_jsonl(dataset: Dataset, file_path: str) -> None:
       
        logger.info(f"Saving dataset to JSONL: {file_path}")
        
        # Create directory if needed
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for example in dataset:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Saved {len(dataset)} examples to JSONL")
    
    @staticmethod
    def save_to_json(dataset: Dataset, file_path: str) -> None:
      
        logger.info(f"Saving dataset to JSON: {file_path}")
        
        # Create directory if needed
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = [example for example in dataset]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(dataset)} examples to JSON")
    
    @staticmethod
    def convert_format(
        input_file: str,
        output_file: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
    ) -> None:
       
        # Load dataset
        dataset = DatasetLoader.load_from_file(input_file, input_format)
        
        # Determine output format
        if output_format is None:
            output_format = Path(output_file).suffix.lstrip('.').lower()
        
        # Save dataset
        if output_format == 'jsonl':
            DatasetLoader.save_to_jsonl(dataset, output_file)
        elif output_format == 'json':
            DatasetLoader.save_to_json(dataset, output_file)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        logger.info(f"Converted {input_file} to {output_file}")


if __name__ == "__main__":
    dataset = DatasetLoader.load_from_hub(
        "nisaar/Articles_Constitution_3300_Instruction_Set",
        split="train"
    )
    print(dataset)
    print(dataset[0])  # see one example
