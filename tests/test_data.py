"""Tests for data processing modules."""
import pytest
from datasets import Dataset

from data.data_processor import DataProcessor
from data.prompt_templates import get_prompt_template, list_templates
from data.dataset_loader import DatasetLoader


class TestPromptTemplates:
    """Test prompt template functionality."""
    
    def test_list_templates(self):
        """Test listing available templates."""
        templates = list_templates()
        assert isinstance(templates, list)
        assert len(templates) > 0
        assert "legal_assistant" in templates
    
    def test_get_template(self):
        """Test getting a template."""
        template = get_prompt_template("legal_assistant")
        assert template is not None
        
        # Test required fields
        required = template.get_required_fields()
        assert "question" in required
        assert "answer" in required
    
    def test_template_format(self):
        """Test template formatting."""
        template = get_prompt_template("legal_assistant")
        
        example = {
            "question": "What is Article 21?",
            "answer": "Article 21 protects life and personal liberty.",
        }
        
        formatted = template.format(example)
        assert isinstance(formatted, str)
        assert "What is Article 21?" in formatted
        assert "Article 21 protects" in formatted
    
    def test_template_format_for_inference(self):
        """Test formatting for inference."""
        template = get_prompt_template("legal_assistant")
        
        prompt = template.format_for_inference("What is PIL?")
        assert isinstance(prompt, str)
        assert "What is PIL?" in prompt
        assert "<|assistant|>" in prompt or "Response:" in prompt


class TestDatasetLoader:
    """Test dataset loading functionality."""
    
    def test_load_from_list(self, tmp_path):
        """Test loading from JSONL file."""
        # Create test file
        test_file = tmp_path / "test.jsonl"
        test_data = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]
        
        with open(test_file, 'w') as f:
            import json
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        # Load dataset
        dataset = DatasetLoader.load_from_jsonl(str(test_file))
        
        assert len(dataset) == 2
        assert dataset[0]["question"] == "Q1"
        assert dataset[1]["answer"] == "A2"
    
    def test_save_to_jsonl(self, tmp_path):
        """Test saving to JSONL."""
        # Create dataset
        data = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]
        dataset = Dataset.from_list(data)
        
        # Save
        output_file = tmp_path / "output.jsonl"
        DatasetLoader.save_to_jsonl(dataset, str(output_file))
        
        # Load and verify
        loaded = DatasetLoader.load_from_jsonl(str(output_file))
        assert len(loaded) == len(dataset)


class TestDataProcessor:
    """Test data processor functionality."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def test_initialization(self, mock_tokenizer):
        """Test processor initialization."""
        processor = DataProcessor(
            tokenizer=mock_tokenizer,
            max_length=512,
        )
        
        assert processor.tokenizer is not None
        assert processor.max_length == 512
    
    def test_format_prompt(self, mock_tokenizer):
        """Test prompt formatting."""
        processor = DataProcessor(
            tokenizer=mock_tokenizer,
            max_length=512,
        )
        
        example = {
            "question": "What is PIL?",
            "answer": "Public Interest Litigation.",
        }
        
        prompt = processor.format_prompt(example)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
    
    def test_tokenize_function(self, mock_tokenizer):
        """Test tokenization."""
        processor = DataProcessor(
            tokenizer=mock_tokenizer,
            max_length=512,
        )
        
        example = {
            "question": "Test question",
            "answer": "Test answer",
        }
        
        tokenized = processor.tokenize_function(example)
        
        assert "input_ids" in tokenized
        assert "attention_mask" in tokenized
        assert "labels" in tokenized
        assert len(tokenized["input_ids"]) <= 512
    
    def test_validate_dataset(self, mock_tokenizer):
        """Test dataset validation."""
        processor = DataProcessor(
            tokenizer=mock_tokenizer,
            max_length=512,
        )
        
        # Valid dataset
        valid_data = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]
        valid_dataset = Dataset.from_list(valid_data)
        
        assert processor.validate_dataset(valid_dataset) is True
        
        # Invalid dataset - missing fields
        invalid_data = [
            {"question": "Q1"},  # Missing answer
        ]
        invalid_dataset = Dataset.from_list(invalid_data)
        
        with pytest.raises(ValueError):
            processor.validate_dataset(invalid_dataset)