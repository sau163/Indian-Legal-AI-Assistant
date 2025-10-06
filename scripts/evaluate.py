"""Evaluation script for Legal AI Assistant."""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import GenerationConfig
from inference.predictor import LegalAIPredictor
from utils.logging_utils import setup_logging
from utils.metrics import MetricsCalculator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Legal AI Assistant"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model or HuggingFace Hub ID",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path to test file (JSONL format with question/answer pairs)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="evaluation_results.json",
        help="Path to output results file",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    
    return parser.parse_args()


def load_test_data(file_path: str, max_examples: int = None) -> List[Dict]:
    """Load test data from file.
    
    Args:
        file_path: Path to test file
        max_examples: Maximum examples to load
        
    Returns:
        List of test examples
    """
    examples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            example = json.loads(line)
            examples.append(example)
            
            if max_examples and len(examples) >= max_examples:
                break
    
    return examples


def evaluate_model(
    predictor: LegalAIPredictor,
    test_data: List[Dict],
    batch_size: int = 1,
) -> Dict:
   
    logger = logging.getLogger(__name__)
    
    predictions = []
    references = []
    questions = []
    
    logger.info(f"Evaluating on {len(test_data)} examples...")
    
    # Generate predictions
    for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
        batch = test_data[i:i + batch_size]
        batch_questions = [ex["question"] for ex in batch]
        
        # Generate answers
        if batch_size == 1:
            answers = [predictor.predict(batch_questions[0])]
        else:
            answers = predictor.predict_batch(batch_questions)
        
        # Collect results
        for example, answer in zip(batch, answers):
            questions.append(example["question"])
            predictions.append(answer)
            references.append(example.get("answer", ""))
    
    # Calculate metrics
    metrics = {
        "num_examples": len(test_data),
        "response_stats": MetricsCalculator.calculate_response_length_stats(predictions),
        "diversity": MetricsCalculator.calculate_diversity_metrics(predictions),
    }
    
    # Store examples
    metrics["examples"] = [
        {
            "question": q,
            "prediction": p,
            "reference": r,
        }
        for q, p, r in zip(questions[:10], predictions[:10], references[:10])
    ]
    
    return metrics


def save_results(results: Dict, output_file: str) -> None:
   
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Results saved to {output_file}")


def print_summary(results: Dict) -> None:
   
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"\nNumber of examples: {results['num_examples']}")
    
    logger.info("\n Response Length Statistics:")
    for key, value in results['response_stats'].items():
        logger.info(f"  {key}: {value:.2f}")
    
    logger.info("\nDiversity Metrics:")
    for key, value in results['diversity'].items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Sample Predictions:")
    logger.info("=" * 80)
    
    for i, example in enumerate(results['examples'][:3], 1):
        logger.info(f"\nExample {i}:")
        logger.info(f"Question: {example['question']}")
        logger.info(f"Prediction: {example['prediction'][:200]}...")
        logger.info("-" * 80)


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Legal AI Assistant - Evaluation")
    logger.info("=" * 80)
    
    # Load test data
    logger.info(f"Loading test data from: {args.test_file}")
    test_data = load_test_data(args.test_file, args.max_examples)
    logger.info(f"Loaded {len(test_data)} test examples")
    
    # Load model
    logger.info(f"Loading model from: {args.model_path}")
    predictor = LegalAIPredictor.from_pretrained(
        model_path=args.model_path,
        generation_config=GenerationConfig(
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.9,
        ),
    )
    logger.info