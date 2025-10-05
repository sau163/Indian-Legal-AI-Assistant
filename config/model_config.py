"""Model configuration for Legal AI Assistant."""
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class ModelType(Enum):
    """Supported model types."""
    FALCON_7B = "tiiuae/falcon-7b-instruct"
    LLAMA2_7B = "meta-llama/Llama-2-7b-chat-hf"
    MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.2"
    GEMMA_7B = "google/gemma-7b-it"
    PHI3_MINI = "microsoft/Phi-3-mini-4k-instruct"


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    # Model-specific target modules
    MODEL_TARGET_MODULES = {
        ModelType.FALCON_7B: ["query_key_value"],
        ModelType.LLAMA2_7B: ["q_proj", "v_proj", "k_proj", "o_proj"],
        ModelType.MISTRAL_7B: ["q_proj", "v_proj", "k_proj", "o_proj"],
        ModelType.GEMMA_7B: ["q_proj", "v_proj", "k_proj", "o_proj"],
        ModelType.PHI3_MINI: ["qkv_proj", "o_proj"],
    }
    
    def get_target_modules(self, model_type: ModelType) -> List[str]:
        """Get target modules for specific model."""
        return self.MODEL_TARGET_MODULES.get(model_type, self.target_modules)


@dataclass
class ModelConfig:
    """Main model configuration."""
    model_type: ModelType = ModelType.MISTRAL_7B
    model_name: Optional[str] = None
    cache_dir: str = "./model_cache"
    device_map: str = "auto"
    trust_remote_code: bool = True
    use_flash_attention: bool = False
    
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    
    def __post_init__(self):
        """Set model name from type if not provided."""
        if self.model_name is None:
            self.model_name = self.model_type.value


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    num_return_sequences: int = 1
    do_sample: bool = True
    early_stopping: bool = True
    num_beams: int = 1
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "num_return_sequences": self.num_return_sequences,
            "do_sample": self.do_sample,
            "early_stopping": self.early_stopping,
            "num_beams": self.num_beams,
        }