"""Example inference script for Legal AI Assistant."""
import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import GenerationConfig
from inference.predictor import LegalAIPredictor
from utils.logging_utils import setup_logging


def main():
    """Run inference examples."""
    parser = argparse.ArgumentParser(
        description="Run inference with Legal AI Assistant"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model or HuggingFace Hub ID",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question to ask (interactive mode if not provided)",
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
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Legal AI Assistant - Inference")
    logger.info("=" * 80)
    
    # Load model
    logger.info(f"Loading model from: {args.model_path}")
    
    predictor = LegalAIPredictor.from_pretrained(
        model_path=args.model_path,
        generation_config=GenerationConfig(
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.9,
            repetition_penalty=1.1,
        ),
    )
    
    logger.info("Model loaded successfully!")
    
    # Single question mode
    if args.question:
        logger.info(f"\nQuestion: {args.question}")
        answer = predictor.predict(args.question)
        print("\n" + "=" * 80)
        print("ANSWER:")
        print("=" * 80)
        print(answer)
        print("=" * 80)
        return
    
    # Interactive mode
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("Type your legal questions below.")
    print("Commands:")
    print("  - 'quit' or 'exit' to exit")
    print("  - 'examples' to see example questions")
    print("=" * 80 + "\n")
    
    # Example questions
    examples = [
        "What are the fundamental rights under Article 21 of the Indian Constitution?",
        "Explain the concept of Public Interest Litigation (PIL) in Indian law.",
        "What is the procedure for filing a First Information Report (FIR)?",
        "Can you summarize the main points of the Right to Information Act, 2005?",
        "What are the grounds for divorce under Hindu Marriage Act, 1955?",
        "Explain the concept of 'Separation of Powers' in the Indian Constitution.",
        "What is the difference between bail and anticipatory bail?",
        "What are the key provisions of the Consumer Protection Act, 2019?",
    ]
    
    while True:
        try:
            # Get user input
            question = input("\n📋 Your question: ").strip()
            
            if not question:
                continue
            
            # Check for commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if question.lower() == 'examples':
                print("\n" + "=" * 80)
                print("EXAMPLE QUESTIONS:")
                print("=" * 80)
                for i, example in enumerate(examples, 1):
                    print(f"{i}. {example}")
                print("=" * 80)
                continue
            
            # Generate answer
            print("\n⏳ Generating answer...")
            answer = predictor.predict(question)
            
            # Display answer
            print("\n" + "=" * 80)
            print("📖 ANSWER:")
            print("=" * 80)
            print(answer)
            print("=" * 80)
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()