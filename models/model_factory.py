
import logging
import os
import torch
from typing import Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from config.model_config import ModelConfig

logger = logging.getLogger(__name__)


class ModelFactory:
    
    @staticmethod
    def create_quantization_config(config: ModelConfig) -> BitsAndBytesConfig:
      
        quant_config = config.quantization
        
        compute_dtype = getattr(torch, quant_config.bnb_4bit_compute_dtype)
        
        return BitsAndBytesConfig(
            load_in_4bit=quant_config.load_in_4bit,
            load_in_8bit=quant_config.load_in_8bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
        )
    
    @staticmethod
    def create_lora_config(config: ModelConfig) -> LoraConfig:
       
        lora_config = config.lora
        target_modules = lora_config.get_target_modules(config.model_type)
        
        return LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            task_type=lora_config.task_type,
        )
    
    @staticmethod
    def load_base_model(
        config: ModelConfig,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
     
        logger.info(f"Loading base model: {config.model_name}")
        
        # Create quantization config
        bnb_config = ModelFactory.create_quantization_config(config)
        
        # Model kwargs
        model_kwargs = {
            "pretrained_model_name_or_path": config.model_name,
            "quantization_config": bnb_config,
            "device_map": config.device_map,
            "trust_remote_code": config.trust_remote_code,
            "torch_dtype": torch.bfloat16,
            "cache_dir": config.cache_dir,
            # Allow using an auth token from environment for gated repos
            "use_auth_token": os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
        }
        
        # Add flash attention if requested and the package is actually installed.
        # Transformers may raise an ImportError during model init if flash_attn
        # isn't installed but an attn implementation is requested. Guard here
        # so training can continue without flash_attn on platforms where it's
        # not available (Windows, or when user hasn't installed it).
        if config.use_flash_attention:
            try:
                # Only enable flash attention 2 if the flash_attn package exists.
                import importlib.util

                if importlib.util.find_spec("flash_attn") is not None:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Using Flash Attention 2")
                else:
                    logger.warning(
                        "Flash Attention requested but package 'flash_attn' not found; continuing without it."
                    )
            except Exception as e:
                logger.warning(f"Flash Attention check failed, continuing without it: {e}")
        
        # If quantization is fully disabled, avoid passing the BitsAndBytes config
        # to Transformers; otherwise Transformers will attempt to use bnb and may
        # trigger CUDA kernels on incompatible GPUs or when bitsandbytes is
        # installed but not desired.
        if not (bnb_config.load_in_4bit or bnb_config.load_in_8bit):
            model_kwargs.pop("quantization_config", None)
            logger.info("Quantization disabled: not passing quantization_config to from_pretrained")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
            cache_dir=config.cache_dir,
            use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
        )
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
        
        logger.info(f"Model loaded successfully: {type(model).__name__}")
        logger.info(f"Model device: {model.device}")
        
        return model, tokenizer
    
    @staticmethod
    def prepare_model_for_training(
        model: PreTrainedModel,
        config: ModelConfig,
    ) -> PreTrainedModel:
       
        logger.info("Preparing model for training...")
        
        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()

        # Prepare for k-bit training only when quantization is enabled.
        # If quantization is disabled (both 4-bit and 8-bit off) skip this
        # step to avoid device-side conversions that may trigger CUDA
        # kernels on incompatible systems.
        if config.quantization.load_in_4bit or config.quantization.load_in_8bit:
            logger.info("Preparing model for k-bit training (quantization enabled)")
            model = prepare_model_for_kbit_training(model)
        else:
            logger.info("Skipping k-bit preparation because quantization is disabled")
        
        # Create and apply LoRA config
        lora_config = ModelFactory.create_lora_config(config)
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        ModelFactory.print_trainable_parameters(model)
        
        return model
    
    @staticmethod
    def load_finetuned_model(
        model_path: str,
        base_config: Optional[ModelConfig] = None,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        
        logger.info(f"Loading fine-tuned model from: {model_path}")
        
        from peft import PeftConfig
        
        # Load PEFT config
        peft_config = PeftConfig.from_pretrained(model_path)
        
        # Create base config if not provided
        if base_config is None:
            base_config = ModelConfig()
        
        # Load base model
        bnb_config = ModelFactory.create_quantization_config(base_config)
        
        model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            quantization_config=bnb_config,
            device_map=base_config.device_map,
            trust_remote_code=base_config.trust_remote_code,
            torch_dtype=torch.bfloat16,
            use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            peft_config.base_model_name_or_path,
            trust_remote_code=base_config.trust_remote_code,
            use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load PEFT model
        model = PeftModel.from_pretrained(
            model,
            model_path,
            use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
        )
        
        logger.info("Fine-tuned model loaded successfully")
        
        return model, tokenizer
    
    @staticmethod
    def print_trainable_parameters(model: PreTrainedModel) -> None:
       
        trainable_params = 0
        all_params = 0
        
        for _, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        trainable_percent = 100 * trainable_params / all_params
        
        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_params:,} || "
            f"Trainable%: {trainable_percent:.4f}%"
        )
    
    @staticmethod
    def save_model(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        output_dir: str,
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
    ) -> None:
     
        logger.info(f"Saving model to {output_dir}")
        
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        if push_to_hub and hub_model_id:
            logger.info(f"Pushing model to Hub: {hub_model_id}")
            model.push_to_hub(hub_model_id, use_auth_token=True)
            tokenizer.push_to_hub(hub_model_id, use_auth_token=True) 