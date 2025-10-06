# Legal AI Assistant - Production Ready

A production-ready, modular implementation of an AI assistant specialized in Indian law, with support for multiple LLM backends and fine-tuning capabilities.

## Features

- **Multiple Model Support**: Falcon-7B, Llama-2-7B, Mistral-7B, Gemma-7B, Phi-3-Mini
- **Efficient Fine-tuning**: QLoRA (4-bit quantization) for memory-efficient training
- **Production API**: FastAPI-based REST API with health checks and batch processing
- **Modular Architecture**: Clean separation of concerns for easy maintenance
- **Docker Support**: Containerized deployment with GPU support
- **Flexible Prompting**: Multiple prompt templates (Legal Assistant, Alpaca, ChatML, Simple Q&A)
- **Comprehensive Logging**: Structured logging for debugging and monitoring
- **Type Safety**: Full type hints and Pydantic validation


## Installation

### Option 1: Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd The_Law_Book

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Docker Installation

```bash
# Build and run with Docker Compose
docker-compose up --build
```

## Usage

### Training

Train a model with default settings:

```bash
python scripts/train.py \
    --model-type mistral \
    --dataset-name /Lawyer_GPT_India \
    --output-dir ./output \
    --num-epochs 3 \
    --batch-size 4
```

Train with custom dataset:

```bash
python scripts/train.py \
    --model-type mistral \
    --dataset-path ./my_dataset.jsonl \
    --dataset-format jsonl \
    --output-dir ./output \
    --num-epochs 5 \
    --learning-rate 2e-4 \
    --lora-r 16
```

Available model types:
- `falcon`: Falcon-7B-Instruct
- `llama2`: Llama-2-7B-Chat
- `mistral`: Mistral-7B-Instruct (recommended)
- `gemma`: Gemma-7B-IT
- `phi3`: Phi-3-Mini

### Inference (Python)

```python
from inference.predictor import LegalAIPredictor
from config.model_config import GenerationConfig

# Load model
predictor = LegalAIPredictor.from_pretrained(
    model_path="./output",  # or HuggingFace Hub ID
    generation_config=GenerationConfig(
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
    )
)

# Generate prediction
question = "What are the fundamental rights under Article 21 of the Indian Constitution?"
answer = predictor.predict(question)
print(answer)
```

### REST API

Start the API server:

```bash
# Set model path (optional, defaults to nisaar/falcon7b-Indian_Law_150Prompts)
export MODEL_PATH="./output"

# Run API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API endpoints:

**Health Check**
```bash
curl http://localhost:8000/health
```

**Single Prediction**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is PIL in Indian law?",
    "temperature": 0.7,
    "max_new_tokens": 256
  }'
```

**Batch Prediction**
```bash
curl -X POST http://localhost:8000/batch-predict \
  -H "Content-Type: application/json" \
  -d '["Question 1?", "Question 2?", "Question 3?"]'
```

Interactive API documentation available at: `http://localhost:8000/docs`

## Dataset Format

Your dataset should be in JSONL format with `question` and `answer` fields:

```jsonl
{"question": "What is the key issue in...", "answer": "The key issue is..."}
{"question": "Can you explain...", "answer": "Certainly! ..."}
```

Alternatively, use `instruction` and `output` fields for Alpaca-style formatting.

## Configuration

### Model Configuration

Edit `config/model_config.py` to customize:
- Model type and quantization settings
- LoRA hyperparameters
- Generation parameters

### Training Configuration

Edit `config/training_config.py` to customize:
- Training hyperparameters
- Batch sizes and gradient accumulation
- Learning rate and scheduler
- Evaluation strategy

## Advanced Features

### Custom Prompt Templates

Add new templates in `data/prompt_templates.py`:

```python
class CustomTemplate(PromptTemplate):
    def format(self, example: Dict) -> str:
        return f"Custom format: {example['question']}"
```

### Multi-GPU Training

```bash
# Use CUDA_VISIBLE_DEVICES to select GPUs
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --model-type mistral
```

### Push to HuggingFace Hub

```bash
python scripts/train.py \
    --model-type mistral \
    --push-to-hub \
    --hub-model-id your-username/your-model-name
```

## Performance Optimization

### Memory Usage

- **4-bit quantization**: Reduces memory by ~75%
- **Gradient checkpointing**: Trades compute for memory
- **LoRA**: Only trains <1% of parameters

### Training Speed

- **Batch size**: Increase if you have GPU memory
- **Gradient accumulation**: Simulate larger batches
- **Mixed precision**: BF16 for faster training
- **Flash Attention**: 2-3x faster on supported GPUs

## Monitoring

### TensorBoard

```bash
tensorboard --logdir=./logs
```

### API Metrics

The API includes:
- Health check endpoint
- Request logging
- Error tracking

## Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=. tests/
```

## Deployment

### Production Deployment

1. **Build Docker image**:
```bash
docker build -t legal-ai-assistant:latest .
```

2. **Run container**:
```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -e MODEL_PATH=your-model-path \
  -v $(pwd)/model_cache:/app/model_cache \
  legal-ai-assistant:latest
```

3. **Use Docker Compose**:
```bash
docker-compose up -d
```

### Kubernetes Deployment

Example deployment manifest included in `k8s/` directory.

## Troubleshooting

### CUDA Out of Memory

- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Reduce `max_seq_length`
- Use smaller model variant

### Slow Training

- Increase `batch_size` if memory allows
- Reduce `logging_steps` and `save_steps`
- Use Flash Attention if available
- Check GPU utilization with `nvidia-smi`

### Model Quality Issues

- Increase training epochs
- Adjust learning rate
- Use larger LoRA rank
- Validate dataset quality

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request


## Citation

If you use this code in your research, please cite:

```bibtex
@software{legal_ai_assistant,
  title={Legal AI Assistant: Production-Ready Indian Law LLM},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/legal-ai-assistant}
}
```

## Acknowledgments

- Dataset: nisaar/Lawyer_GPT_India on HuggingFace
- Base models: Mistral AI, Meta, Google, Microsoft, TII

## License

This project is released under the MIT License. You are free to use, modify,
and distribute the source code, provided that you include the original MIT
license notice in any substantial portions of the software.

Please note the following important licensing and usage considerations:

- Third-party models and datasets referenced by this project (for example,
  models from Hugging Face such as `mistralai/Mistral-7B-Instruct-v0.2`) are governed by their own licenses and access terms. Some models are gated or have usage restrictions — make
  sure you review and comply with the license and terms of service for any
  model or dataset you download or use.

- When pushing fine-tuned models derived from third-party base models to a
  public hub, check whether redistribution of derivative models is permitted
  by the base model's license. If in doubt, keep your model private or contact
  the model authors for permission.

- This repository may include small utility scripts, configuration files, and
  example code meant to help you run experiments. Those portions are covered
  by the MIT License included with this repository.

If you want to use a different license for your derived work, feel free to do
so for your own fork or distribution, but the original project files that are
derived from this repository must continue to include the original MIT license
notice.