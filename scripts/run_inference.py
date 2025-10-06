
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.predictor import LegalAIPredictor
from config.model_config import GenerationConfig, ModelConfig
from models.model_factory import ModelFactory


def parse_args():
    p = argparse.ArgumentParser("Run inference with LegalAIPredictor")
    p.add_argument("--model-path", type=str, default="./output", help="Path to fine-tuned model or HF hub id")
    p.add_argument("--question", type=str, required=True, help="Question text to ask the model")
    p.add_argument("--device", type=str, default=None, help="Device to run on (e.g. cpu or cuda:0). Defaults to model device or cpu")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not args.model_path:
        logger.error("Please provide --model-path pointing to a local model directory or a HuggingFace Hub id.")
        return

    # Load predictor
    # First try loading a PEFT/adapter-style fine-tuned model (the common case
    # when using LoRA). If that fails because the directory doesn't contain a
    # PEFT adapter, fall back to loading a base pretrained model from the same
    # path or hub id.
    try:
        predictor = LegalAIPredictor.from_pretrained(
            model_path=args.model_path,
            generation_config=GenerationConfig(
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            ),
        )
    except Exception as e:
        # Detect the "adapter_config.json" missing error from PEFT and try base model
        msg = str(e)
        logger.warning(f"PEFT/adapter load failed: {msg}")

        try:
            logger.info("Attempting to load as a base pretrained model (no PEFT adapter)...")

            # Build a minimal ModelConfig that points to the provided model_path
            base_cfg = ModelConfig()
            base_cfg.model_name = args.model_path
            # Respect explicit device request; default to CPU to avoid CUDA kernel
            # issues on systems where the installed PyTorch may not support the
            # GPU. If the user requested a GPU device explicitly (e.g. --device cuda:0)
            # set that instead.
            if args.device:
                base_cfg.device_map = args.device
            else:
                base_cfg.device_map = "cpu"

            model, tokenizer = ModelFactory.load_base_model(base_cfg)

            predictor = LegalAIPredictor(
                model=model,
                tokenizer=tokenizer,
                generation_config=GenerationConfig(
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                ),
            )

        except Exception as e2:
            logger.error(f"Failed to load model as base pretrained model: {e2}")
            logger.error(
                "Unable to load model. Make sure './output' contains either a PEFT adapter (adapter_config.json) "
                "or a full model saved with `model.save_pretrained(...)` and `tokenizer.save_pretrained(...)`, "
                "or pass a HuggingFace Hub model id. If the model is gated, set HUGGINGFACE_HUB_TOKEN in your env."
            )
            raise

    # If device override provided, set predictor device
    if args.device:
        predictor.device = args.device

    # Run prediction
    try:
        answer = predictor.predict(
            args.question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print("\n=== Question ===")
        print(args.question)
        print("\n=== Answer ===")
        print(answer)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()
