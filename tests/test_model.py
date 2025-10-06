"""Tests for model modules."""
import pytest
import torch

from config.model_config import ModelConfig, ModelType, LoRAConfig, QuantizationConfig
from models.model_factory import ModelFactory


class TestModelConfig:
    """Test model configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ModelConfig()
        
        assert config.model_type == ModelType.MISTRAL_7B
        assert config.device_map == "auto"
        assert config.trust_remote_code is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ModelConfig(
            model_type=ModelType.LLAMA2_7B,
            device_map="cuda:0",
        )
        
        assert config.model_type == ModelType.LLAMA2_7B
        assert config.device_map == "cuda:0"
    
    def test_lora_config(self):
        """Test LoRA configuration."""
        lora = LoRAConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
        )
        
        assert lora.r == 32
        assert lora.lora_alpha == 64
        assert lora.lora_dropout == 0.1
    
    def test_lora_target_modules(self):
        """Test getting target modules for different models."""
        lora = LoRAConfig()
        
        # Test Falcon
        falcon_modules = lora.get_target_modules(ModelType.FALCON_7B)
        assert "query_key_value" in falcon_modules
        
        # Test Mistral
        mistral_modules = lora.get_target_modules(ModelType.MISTRAL_7B)
        assert "q_proj" in mistral_modules
        assert "v_proj" in mistral_modules


class TestQuantizationConfig:
    """Test quantization configuration."""
    
    def test_default_quantization(self):
        """Test default quantization settings."""
        config = QuantizationConfig()
        
        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_use_double_quant is True


class TestModelFactory:
    """Test model factory."""
    
    def test_create_quantization_config(self):
        """Test creating quantization config."""
        model_config = ModelConfig()
        
        bnb_config = ModelFactory.create_quantization_config(model_config)
        
        assert bnb_config is not None
        assert hasattr(bnb_config, 'load_in_4bit')
    
    def test_create_lora_config(self):
        """Test creating LoRA config."""
        model_config = ModelConfig()
        
        lora_config = ModelFactory.create_lora_config(model_config)
        
        assert lora_config is not None
        assert lora_config.r > 0
        assert lora_config.lora_alpha > 0
        assert len(lora_config.target_modules) > 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_print_trainable_parameters(self):
        """Test printing trainable parameters."""
        # Create a simple mock model
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        
        # Should not raise any errors
        ModelFactory.print_trainable_parameters(model)


class TestGenerationConfig:
    """Test generation configuration."""
    
    def test_default_generation_config(self):
        """Test default generation settings."""
        from config.model_config import GenerationConfig
        
        config = GenerationConfig()
        
        assert config.max_new_tokens == 512
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.do_sample is True
    
    def test_custom_generation_config(self):
        """Test custom generation settings."""
        from config.model_config import GenerationConfig
        
        config = GenerationConfig(
            max_new_tokens=256,
            temperature=0.5,
            top_p=0.95,
        )
        
        assert config.max_new_tokens == 256
        assert config.temperature == 0.5
        assert config.top_p == 0.95
    
    def test_generation_config_to_dict(self):
        """Test converting config to dict."""
        from config.model_config import GenerationConfig
        
        config = GenerationConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "max_new_tokens" in config_dict
        assert "temperature" in config_dict