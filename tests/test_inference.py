"""Tests for inference modules."""
import pytest

from inference.response_formatter import ResponseFormatter, CitationExtractor


class TestResponseFormatter:
    """Test response formatting."""
    
    def test_remove_special_tokens(self):
        """Test removing special tokens."""
        text = "Hello <|endoftext|> World <pad> Test"
        cleaned = ResponseFormatter.remove_special_tokens(text)
        
        assert "<|endoftext|>" not in cleaned
        assert "<pad>" not in cleaned
        assert "Hello" in cleaned
        assert "World" in cleaned
    
    def test_fix_spacing(self):
        """Test fixing spacing issues."""
        text = "Hello  ,  World  .Test"
        fixed = ResponseFormatter.fix_spacing(text)
        
        assert "  " not in fixed  # No double spaces
        assert " ," not in fixed  # No space before comma
        assert " ." not in fixed  # No space before period
    
    def test_truncate_at_sentence(self):
        """Test truncating at sentence boundary."""
        text = "First sentence. Second sentence. Third sentence."
        
        # Truncate at 30 chars (within second sentence)
        truncated = ResponseFormatter.truncate_at_sentence(text, max_length=30)
        
        assert truncated.endswith(".")
        assert len(truncated) <= 30
        assert "First sentence" in truncated
    
    def test_extract_citations(self):
        """Test extracting legal citations."""
        text = """
        According to AIR 1973 SC 1461, the basic structure cannot be amended.
        See also 2020 (5) SCC 1 for more details.
        """
        
        citations = ResponseFormatter.extract_citations(text)
        
        assert len(citations) > 0
        assert any("AIR" in c for c in citations)
    
    def test_clean_response(self):
        """Test complete response cleaning."""
        text = "  Hello  <|endoftext|>  World  .  Test  "
        cleaned = ResponseFormatter.clean_response(text)
        
        assert cleaned.startswith("Hello")
        assert cleaned.endswith("Test")
        assert "<|endoftext|>" not in cleaned
    
    def test_format_as_structured_response(self):
        """Test structured response formatting."""
        question = "What is PIL?"
        answer = "PIL is Public Interest Litigation as per AIR 1981 SC 149."
        
        response = ResponseFormatter.format_as_structured_response(
            question, answer
        )
        
        assert isinstance(response, dict)
        assert response["question"] == question
        assert "answer" in response
        assert "citations" in response
        assert "length" in response


class TestCitationExtractor:
    """Test citation extraction."""
    
    def test_parse_air_citation(self):
        """Test parsing AIR citation."""
        citation = "AIR 1973 SC 1461"
        parsed = CitationExtractor.parse_citation(citation)
        
        assert parsed is not None
        assert parsed["year"] == "1973"
        assert parsed["court"] == "SC"
        assert parsed["number"] == "1461"
        assert parsed["reporter"] == "AIR"
    
    def test_parse_scc_citation(self):
        """Test parsing SCC citation."""
        citation = "2020 (5) SCC 1"
        parsed = CitationExtractor.parse_citation(citation)
        
        assert parsed is not None
        assert parsed["year"] == "2020"
        assert parsed["volume"] == "5"
        assert parsed["reporter"] == "SCC"
        assert parsed["number"] == "1"
    
    def test_format_citation(self):
        """Test formatting citation."""
        citation_dict = {
            "year": "1973",
            "court": "SC",
            "number": "1461",
            "reporter": "AIR",
        }
        
        formatted = CitationExtractor.format_citation(citation_dict)
        
        assert "AIR" in formatted
        assert "1973" in formatted
        assert "SC" in formatted
        assert "1461" in formatted


class TestPredictorMock:
    """Test predictor with mock components."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def test_extract_answer(self, mock_tokenizer):
        """Test answer extraction."""
        from inference.predictor import LegalAIPredictor
        
        # Create mock model
        class MockModel:
            def __init__(self):
                self.config = type('obj', (object,), {'use_cache': False})()
                self.device = "cpu"
            
            def eval(self):
                pass
        
        model = MockModel()
        
        predictor = LegalAIPredictor(
            model=model,
            tokenizer=mock_tokenizer,
        )
        
        # Test with assistant marker
        response = "Question text <|assistant|> This is the answer"
        prompt = "Question text <|assistant|>"
        
        answer = predictor._extract_answer(response, prompt)
        
        assert "This is the answer" in answer
        assert "<|assistant|>" not in answer