import os
import uuid
import base64
import logging
import httpx
from typing import Dict, Any
from app.core.config import settings
from app.ai.image_base import ImageProvider

logger = logging.getLogger(__name__)

class GeminiImageProvider(ImageProvider):
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        self.model = "imagen-3.0-generate-002"
        self.static_dir = os.path.join(os.getcwd(), "static", "generated_images")
        
        # Ensure static directory exists
        os.makedirs(self.static_dir, exist_ok=True)

    def generate_reference_image(self, concept_prompt: str, style: str, size: str = "1024x1024") -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateImages?key={self.api_key}"
        
        # Parse aspect ratio from size if needed. Defaults to 1:1.
        aspect_ratio = "1:1"
        if size == "16:9" or "1920" in size:
            aspect_ratio = "16:9"
        elif size == "9:16":
            aspect_ratio = "9:16"
        elif size == "4:3":
            aspect_ratio = "4:3"
            
        payload = {
            "prompt": concept_prompt,
            "numberOfImages": 1,
            "outputMimeType": "image/jpeg",
            "aspectRatio": aspect_ratio
        }
        
        logger.info(f"Generating Imagen reference image for prompt: {concept_prompt[:60]}...")
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload)
            if response.status_code != 200:
                logger.error(f"Imagen API failed with status {response.status_code}: {response.text}")
                raise ValueError(f"Imagen API error: {response.text}")
                
            res_data = response.json()
            images = res_data.get("generatedImages", [])
            if not images:
                raise ValueError("No images returned from Imagen API.")
                
            image_bytes_b64 = images[0]["image"]["imageBytes"]
            image_bytes = base64.b64decode(image_bytes_b64)
            
            # Save file locally
            filename = f"{uuid.uuid4()}.jpg"
            filepath = os.path.join(self.static_dir, filename)
            with open(filepath, "wb") as f:
                f.write(image_bytes)
                
            # Return relative path URL
            image_url = f"/static/generated_images/{filename}"
            
            return {
                "image_url": image_url,
                "prompt": concept_prompt,
                "provider": "gemini_image",
                "reference_only": True,
                "notice": "Reference only - final artwork belongs to the designer."
            }

    def generate_variations(self, image_url: str, instruction: str) -> Dict[str, Any]:
        # Variations can be simple prompts using image context, 
        # for MVP we can rerun with a prompt modification.
        modified_prompt = f"Reference variation of {image_url}: {instruction}"
        return self.generate_reference_image(modified_prompt, style="custom")

    def test_connection(self) -> bool:
        try:
            # Generate a tiny image to test connection
            self.generate_reference_image("A gray background.", style="test", size="1:1")
            return True
        except Exception as e:
            logger.error(f"Gemini Image connection test failed: {str(e)}")
            return False
