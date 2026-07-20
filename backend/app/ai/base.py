from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class LLMProvider(ABC):
    @abstractmethod
    def extract_brief(self, raw_imagination: str, style_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract a structured creative brief from raw imagination."""
        pass

    @abstractmethod
    def generate_questions(self, brief: Dict[str, Any]) -> List[str]:
        """Generate 5-8 clarifying questions based on the brief."""
        pass

    @abstractmethod
    def generate_concepts(self, brief: Dict[str, Any], answers: List[Dict[str, Any]], style_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate exactly 10 concepts, one per style category."""
        pass

    @abstractmethod
    def regenerate_concept(self, brief: Dict[str, Any], original_concept: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        """Regenerate a concept using custom instructions."""
        pass

    @abstractmethod
    def combine_concepts(self, brief: Dict[str, Any], concept_a: Dict[str, Any], concept_b: Dict[str, Any], custom_instruction: Optional[str] = None) -> Dict[str, Any]:
        """Combine two concepts into one new hybrid concept."""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test API key/connection validity."""
        pass
