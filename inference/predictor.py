"""Inference module for Legal AI Assistant."""
import logging
import torch
from typing import List, Optional, Dict
from transformers import PreTrainedModel, PreTrainedTokenizer

from config.model_config import GenerationConfig, ModelConfig
from models.model_factory import ModelFactory
from data.prompt_templates import get_prompt_template

logger = logging.getLogger(__name__)


class LegalAIPredictor:
    """Predictor for Legal AI Assistant."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        generation_config: Optional[GenerationConfig] = None,
        prompt_template: str = "legal_assistant",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
      
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config or GenerationConfig()
        self.prompt_template = get_prompt_template(prompt_template)
        self.device = device
        
        # Move model to device
        if hasattr(self.model, 'device'):
            self.device = self.model.device
        
        # Set model to eval mode
        self.model.eval()
        
        # Enable caching for faster generation
        self.model.config.use_cache = True
        
        logger.info(f"Predictor initialized on device: {self.device}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        model_config: Optional[ModelConfig] = None,
        generation_config: Optional[GenerationConfig] = None,
        prompt_template: str = "legal_assistant",
    ) -> "LegalAIPredictor":
        
        logger.info(f"Loading predictor from: {model_path}")
        
        model, tokenizer = ModelFactory.load_finetuned_model(
            model_path,
            model_config,
        )
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            prompt_template=prompt_template,
        )
    
    def predict(
        self,
        question: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> str:
       
        # Format prompt
        prompt = self.prompt_template.format_for_inference(question)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)
        
        # Update generation config with provided parameters
        gen_config = self.generation_config.to_dict()
        if max_new_tokens is not None:
            gen_config["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            gen_config["temperature"] = temperature
        if top_p is not None:
            gen_config["top_p"] = top_p
        gen_config.update(kwargs)
        
        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **gen_config,
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )
        
        # Extract answer from response
        answer = self._extract_answer(response, prompt)
        
        return answer
    
    def predict_batch(
        self,
        questions: List[str],
        batch_size: int = 8,
        **kwargs,
    ) -> List[str]:
      
        answers = []
        
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            
            # Process batch
            batch_answers = [
                self.predict(q, **kwargs)
                for q in batch_questions
            ]
            
            answers.extend(batch_answers)
        
        return answers
    
    def _extract_answer(self, response: str, prompt: str) -> str:
        
        # Try to find assistant marker
        assistant_markers = [
            "<|assistant|>",
            "### Response:",
            "Answer:",
            "<|im_start|>assistant",
        ]
        
        for marker in assistant_markers:
            if marker in response:
                parts = response.split(marker)
                if len(parts) > 1:
                    answer = parts[-1].strip()
                    # Remove end markers if present
                    end_markers = ["<|im_end|>", "<|endoftext|>"]
                    for end_marker in end_markers:
                        answer = answer.replace(end_marker, "")
                    return answer.strip()
        
        # Fallback: remove prompt from response
        if prompt in response:
            answer = response.replace(prompt, "").strip()
            return answer
        
        # Last resort: return full response
        return response.strip()
    
    def chat(
        self,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> str:
      
        if conversation_history is None:
            conversation_history = []
        
        # Build prompt from conversation history
        prompt_parts = []
        
        for message in conversation_history:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "user":
                prompt_parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}")
        
        # Add final assistant prompt
        prompt_parts.append("<|assistant|>")
        
        full_prompt = "\n".join(prompt_parts)
        
        # Generate
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)
        
        gen_config = self.generation_config.to_dict()
        gen_config.update(kwargs)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **gen_config,
            )
        
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )
        
        answer = self._extract_answer(response, full_prompt)
        
        return answer