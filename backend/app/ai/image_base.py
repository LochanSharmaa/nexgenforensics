from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class ImageProvider(ABC):
    @abstractmethod
    def generate_reference_image(self, concept_prompt: str, style: str, size: str = "1024x1024") -> Dict[str, Any]:
        """
        Generate a reference-only image.
        Returns:
            dict containing:
            {
              "image_url": str,
              "prompt": str,
              "provider": str,
              "reference_only": True,
              "notice": "Reference only - final artwork belongs to the designer."
            }
        """
        pass

    @abstractmethod
    def generate_variations(self, image_url: str, instruction: str) -> Dict[str, Any]:
        """Generate variations of the image based on instruction."""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test API key/connection validity."""
        pass
