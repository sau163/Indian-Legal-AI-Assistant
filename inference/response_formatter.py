
import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ResponseFormatter:
   
    
    @staticmethod
    def clean_response(response: str) -> str:
       
        # Remove special tokens
        response = ResponseFormatter.remove_special_tokens(response)
        
        # Fix spacing
        response = ResponseFormatter.fix_spacing(response)
        
        # Remove repetitions
        response = ResponseFormatter.remove_repetitions(response)
        
        # Clean formatting
        response = response.strip()
        
        return response
    
    @staticmethod
    def remove_special_tokens(text: str) -> str:
      
        special_tokens = [
            "<|endoftext|>",
            "<|im_end|>",
            "<|im_start|>",
            "<pad>",
            "</s>",
            "<s>",
            "[PAD]",
            "[CLS]",
            "[SEP]",
        ]
        
        for token in special_tokens:
            text = text.replace(token, "")
        
        return text
    
    @staticmethod
    def fix_spacing(text: str) -> str:
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove spaces before punctuation
        text = re.sub(r' ([.,!?;:])', r'\1', text)
        
        # Add space after punctuation if missing
        text = re.sub(r'([.,!?;:])([A-Z])', r'\1 \2', text)
        
        # Fix multiple newlines
        text = re.sub(r'\n+', '\n\n', text)
        
        return text
    
    @staticmethod
    def remove_repetitions(text: str, max_repeats: int = 3) -> str:
       
        sentences = text.split('. ')
        
        # Remove duplicate consecutive sentences
        cleaned_sentences = []
        prev_sentence = None
        repeat_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if sentence == prev_sentence:
                repeat_count += 1
                if repeat_count < max_repeats:
                    cleaned_sentences.append(sentence)
            else:
                cleaned_sentences.append(sentence)
                repeat_count = 0
            
            prev_sentence = sentence
        
        return '. '.join(cleaned_sentences)
    
    @staticmethod
    def truncate_at_sentence(
        text: str,
        max_length: Optional[int] = None,
    ) -> str:
      
        if max_length is None or len(text) <= max_length:
            return text
        
        # Find last sentence boundary before max_length
        truncated = text[:max_length]
        
        # Find last period, question mark, or exclamation
        last_punct = max(
            truncated.rfind('.'),
            truncated.rfind('?'),
            truncated.rfind('!'),
        )
        
        if last_punct > 0:
            return truncated[:last_punct + 1].strip()
        
        return truncated.strip() + "..."
    
    @staticmethod
    def extract_citations(text: str) -> List[str]:
      
        # Pattern for Indian legal citations
        # E.g., "AIR 1973 SC 1461", "2020 (5) SCC 1"
        patterns = [
            r'\b\d{4}\s*\(\d+\)\s*[A-Z]+\s*\d+\b',  # 2020 (5) SCC 1
            r'\bAIR\s*\d{4}\s*[A-Z]+\s*\d+\b',      # AIR 1973 SC 1461
            r'\b\d{4}\s*[A-Z]+\s*\d+\b',            # 1973 SC 1461
        ]
        
        citations = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        
        return list(set(citations))  # Remove duplicates
    
    @staticmethod
    def format_as_markdown(text: str, title: Optional[str] = None) -> str:
       
        lines = []
        
        if title:
            lines.append(f"# {title}\n")
        
        lines.append(text)
        
        return '\n'.join(lines)
    
    @staticmethod
    def format_as_structured_response(
        question: str,
        answer: str,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, any]:
       
        response = {
            "question": question,
            "answer": ResponseFormatter.clean_response(answer),
            "citations": ResponseFormatter.extract_citations(answer),
            "length": len(answer.split()),
        }
        
        if metadata:
            response["metadata"] = metadata
        
        return response
    
    @staticmethod
    def highlight_legal_terms(text: str) -> str:
       
        # Common Indian legal terms
        legal_terms = [
            "Article",
            "Constitution",
            "Supreme Court",
            "High Court",
            "Parliament",
            "Legislature",
            "Act",
            "Section",
            "Clause",
            "Fundamental Rights",
            "Directive Principles",
            "PIL",
            "FIR",
            "IPC",
            "CrPC",
            "CPC",
        ]
        
        for term in legal_terms:
            # Use word boundary to match whole words only
            pattern = r'\b' + re.escape(term) + r'\b'
            text = re.sub(
                pattern,
                f"**{term}**",
                text,
                flags=re.IGNORECASE,
            )
        
        return text


class CitationExtractor:
    """Extract and format legal citations."""
    
    @staticmethod
    def parse_citation(citation: str) -> Optional[Dict[str, str]]:
       
        # Try different patterns
        
        # Pattern: AIR YEAR COURT NUMBER
        match = re.match(r'AIR\s*(\d{4})\s*([A-Z]+)\s*(\d+)', citation)
        if match:
            return {
                "year": match.group(1),
                "court": match.group(2),
                "number": match.group(3),
                "reporter": "AIR",
            }
        
        # Pattern: YEAR (VOLUME) REPORTER NUMBER
        match = re.match(r'(\d{4})\s*\((\d+)\)\s*([A-Z]+)\s*(\d+)', citation)
        if match:
            return {
                "year": match.group(1),
                "volume": match.group(2),
                "reporter": match.group(3),
                "number": match.group(4),
            }
        
        return None
    
    @staticmethod
    def format_citation(citation_dict: Dict[str, str]) -> str:
       
        if "reporter" in citation_dict and citation_dict["reporter"] == "AIR":
            return (
                f"AIR {citation_dict['year']} "
                f"{citation_dict['court']} {citation_dict['number']}"
            )
        
        if "volume" in citation_dict:
            return (
                f"{citation_dict['year']} ({citation_dict['volume']}) "
                f"{citation_dict['reporter']} {citation_dict['number']}"
            )
        
        return str(citation_dict)