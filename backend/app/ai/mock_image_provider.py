import time
from typing import Dict, Any
from app.ai.image_base import ImageProvider

class MockImageProvider(ImageProvider):
    # Distinct Unsplash URLs for each style to make the mock run look amazing
    STYLE_IMAGE_MAP = {
        "minimal luxury": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?q=80&w=600&auto=format&fit=crop",
        "cinematic dramatic": "https://images.unsplash.com/photo-1478760329108-5c3ed9d495a0?q=80&w=600&auto=format&fit=crop",
        "futuristic premium": "https://images.unsplash.com/photo-1544383835-bda2bc66a55d?q=80&w=600&auto=format&fit=crop",
        "surreal dreamlike": "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?q=80&w=600&auto=format&fit=crop",
        "editorial magazine": "https://images.unsplash.com/photo-1534447677768-be436bb09401?q=80&w=600&auto=format&fit=crop",
        "product advertisement": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?q=80&w=600&auto=format&fit=crop",
        "abstract artistic": "https://images.unsplash.com/photo-1541701494587-cb58502866ab?q=80&w=600&auto=format&fit=crop",
        "dark premium": "https://images.unsplash.com/photo-1507608869274-d3177c8bb4c7?q=80&w=600&auto=format&fit=crop",
        "human emotional story": "https://images.unsplash.com/photo-1516450360452-9312f5e86fc7?q=80&w=600&auto=format&fit=crop",
        "experimental poster": "https://images.unsplash.com/photo-1550684848-fac1c5b4e853?q=80&w=600&auto=format&fit=crop"
    }

    def generate_reference_image(self, concept_prompt: str, style: str, size: str = "1024x1024") -> Dict[str, Any]:
        time.sleep(1.5)  # Simulate network latency
        
        style_key = style.lower().strip()
        image_url = self.STYLE_IMAGE_MAP.get(
            style_key, 
            "https://images.unsplash.com/photo-1523275335684-37898b6baf30?q=80&w=600&auto=format&fit=crop" # Default
        )

        return {
            "image_url": image_url,
            "prompt": concept_prompt,
            "provider": "mock",
            "reference_only": True,
            "notice": "Reference only - final artwork belongs to the designer."
        }

    def generate_variations(self, image_url: str, instruction: str) -> Dict[str, Any]:
        time.sleep(1.0)
        return {
            "image_url": image_url,
            "prompt": f"Variation based on instruction: {instruction}",
            "provider": "mock",
            "reference_only": True,
            "notice": "Reference only - final artwork belongs to the designer."
        }

    def test_connection(self) -> bool:
        return True
