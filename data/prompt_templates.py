"""Prompt templates for Legal AI Assistant."""
from abc import ABC, abstractmethod
from typing import Dict, List


class PromptTemplate(ABC):
    """Base class for prompt templates."""
    
    @abstractmethod
    def format(self, example: Dict) -> str:
        """Format example into prompt string."""
        pass
    
    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """Get list of required fields in example."""
        pass
    
    @abstractmethod
    def format_for_inference(self, question: str) -> str:
        """Format question for inference (without answer)."""
        pass


class LegalAssistantTemplate(PromptTemplate):
    """Template for legal assistant conversations."""
    
    def format(self, example: Dict) -> str:
        """Format training example with question and answer."""
        question = example.get("question", "")
        answer = example.get("answer", "")
        
        return f"""<|system|>
You are an expert legal assistant specializing in Indian law. Provide accurate, well-reasoned legal information and analysis.
<|user|>
{question}
<|assistant|>
{answer}"""
    
    def format_for_inference(self, question: str) -> str:
        """Format question for inference."""
        return f"""<|system|>
You are an expert legal assistant specializing in Indian law. Provide accurate, well-reasoned legal information and analysis.
<|user|>
{question}
<|assistant|>
"""
    
    def get_required_fields(self) -> List[str]:
        """Get required fields."""
        return ["question", "answer"]


class AlpacaTemplate(PromptTemplate):
    """Alpaca-style prompt template."""
    
    def format(self, example: Dict) -> str:
        """Format training example."""
        instruction = example.get("instruction", example.get("question", ""))
        input_text = example.get("input", "")
        response = example.get("output", example.get("answer", ""))
        
        if input_text:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{response}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""
    
    def format_for_inference(self, question: str, input_text: str = "") -> str:
        """Format question for inference."""
        if input_text:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{question}

### Input:
{input_text}

### Response:
"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{question}

### Response:
"""
    
    def get_required_fields(self) -> List[str]:
        """Get required fields."""
        return ["question", "answer"]  # Flexible field names


class ChatMLTemplate(PromptTemplate):
    """ChatML format template."""
    
    def format(self, example: Dict) -> str:
        """Format training example."""
        question = example.get("question", "")
        answer = example.get("answer", "")
        
        return f"""<|im_start|>system
You are an expert legal assistant specializing in Indian law.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""
    
    def format_for_inference(self, question: str) -> str:
        """Format question for inference."""
        return f"""<|im_start|>system
You are an expert legal assistant specializing in Indian law.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    
    def get_required_fields(self) -> List[str]:
        """Get required fields."""
        return ["question", "answer"]


class SimpleQATemplate(PromptTemplate):
    """Simple Q&A template."""
    
    def format(self, example: Dict) -> str:
        """Format training example."""
        question = example.get("question", "")
        answer = example.get("answer", "")
        
        return f"""Question: {question}

Answer: {answer}"""
    
    def format_for_inference(self, question: str) -> str:
        """Format question for inference."""
        return f"""Question: {question}

Answer:"""
    
    def get_required_fields(self) -> List[str]:
        """Get required fields."""
        return ["question", "answer"]


# Template registry
TEMPLATE_REGISTRY = {
    "legal_assistant": LegalAssistantTemplate(),
    "alpaca": AlpacaTemplate(),
    "chatml": ChatMLTemplate(),
    "simple_qa": SimpleQATemplate(),
}


def get_prompt_template(template_name: str = "legal_assistant") -> PromptTemplate:
   
    if template_name not in TEMPLATE_REGISTRY:
        raise ValueError(
            f"Template '{template_name}' not found. "
            f"Available templates: {list(TEMPLATE_REGISTRY.keys())}"
        )
    
    return TEMPLATE_REGISTRY[template_name]


def list_templates() -> List[str]:
    """List available prompt templates."""
    return list(TEMPLATE_REGISTRY.keys())