import time
from typing import Dict, Any, List, Optional
from app.ai.base import LLMProvider

class MockLLMProvider(LLMProvider):
    def extract_brief(self, raw_imagination: str, style_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        time.sleep(0.5)  # Simulate API latency
        
        # Simple heuristic keywords
        imagination_lower = raw_imagination.lower()
        
        subject = "luxury perfume bottle floating above black water" if "perfume" in imagination_lower else "sleek design concept"
        design_type = "luxury poster advertisement" if "perfume" in imagination_lower else "conceptual artwork direction"
        moods = ["dark", "premium", "cinematic", "mysterious"] if "dark" in imagination_lower or "perfume" in imagination_lower else ["modern", "elegant", "clean"]
        colors = ["blue-black", "silver", "moonlight white"] if "perfume" in imagination_lower else ["charcoal", "accent gold", "pure white"]
        
        if "perfume" not in imagination_lower:
            # Extract basic terms from imagination to populate subject
            words = [w for w in raw_imagination.split() if len(w) > 3][:5]
            if words:
                subject = " ".join(words)
        
        return {
            "main_subject": subject,
            "design_type": design_type,
            "target_audience": "high-end luxury buyers" if "perfume" in imagination_lower else "design enthusiasts and premium consumers",
            "mood": moods,
            "colors": colors,
            "fixed_elements": [subject, "minimal text"],
            "flexible_elements": ["lighting accents", "metaphoric props"],
            "avoid_elements": ["cheap commercial look", "crowded layout", "bright playful colors"]
        }

    def generate_questions(self, brief: Dict[str, Any]) -> List[str]:
        time.sleep(0.5)
        subject = brief.get("main_subject", "design object")
        return [
            f"Should the {subject} be the absolute center of focus, or off-center to create negative space?",
            f"What specific type of typography is preferred? A traditional serif, a clean sans-serif, or a modern experimental font?",
            f"Do you want to integrate any natural elements (water ripples, wind, mist) or keep it purely studio-based?",
            f"Is there a secondary product or logo that needs a prominent placeholder, or is a single subject sufficient?",
            f"How prominent should the 'Reference only' notice be on the final mockups?"
        ]

    def generate_concepts(self, brief: Dict[str, Any], answers: List[Dict[str, Any]], style_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        time.sleep(1.0)
        subject = brief.get("main_subject", "subject")
        design_type = brief.get("design_type", "poster")
        moods = ", ".join(brief.get("mood", ["premium"]))
        colors = brief.get("colors", ["blue-black", "silver"])
        color_primary = colors[0] if colors else "black"
        color_secondary = colors[1] if len(colors) > 1 else "white"

        categories = [
            ("Minimal luxury", "A single centered subject with massive negative space."),
            ("Cinematic dramatic", "A low-angle shot with strong backlighting and ambient fog."),
            ("Futuristic premium", "Holographic outlines and glowing neon elements framing the subject."),
            ("Surreal dreamlike", "The subject floating within a giant bubble over an ocean of mercury."),
            ("Editorial magazine", "A split-screen layout with bold, overlapping typography and high-fashion model accents."),
            ("Product advertisement", "A studio-light focus showing fine product details on a marble podium."),
            ("Abstract artistic", "Geometric paint strokes and raw material textures overlapping the subject."),
            ("Dark premium", "High-contrast chiaroscuro lighting on a carbon fiber background."),
            ("Human emotional story", "A designer's hand reaching out towards the floating subject to emphasize craftsmanship."),
            ("Experimental poster", "Distorted layouts, glitch typography, and thermal-imaging style color overlays.")
        ]

        concepts_list = []
        for i, (style, description) in enumerate(categories, 1):
            concepts_list.append({
                "concept_number": i,
                "title": f"{style} Direction - Concept {i}",
                "style_category": style,
                "main_visual_idea": f"{description} featuring {subject} in a {moods} mood.",
                "composition": f"Balanced grid alignment, emphasizing the {style} layout style.",
                "lighting": "Soft rim lighting with premium reflections." if "luxury" in style.lower() else "Dramatic high contrast chiaroscuro.",
                "background": f"Clean dark canvas with subtle gradients of {color_primary} and {color_secondary}.",
                "color_palette": [color_primary, color_secondary, "accent accent"],
                "typography_direction": "Elegant high-contrast serif font, centered at the bottom margin.",
                "creative_twist": "The negative space subtly traces the silhouette of a classic frame.",
                "designer_execution_notes": "Ensure high-contrast boundaries; avoid visual noise so the core concept shines.",
                "reference_image_prompt": f"Reference only: {subject} in {style} style, {moods} environment, {color_primary} and {color_secondary} palette, professional design asset.",
                "reference_only_notice": "Reference only - final artwork belongs to the designer."
            })

        return {
            "project_title": f"Expanded {subject.title()}",
            "brief_summary": f"A comprehensive visual direction exploration for {subject} aimed at luxury marketing.",
            "concepts": concepts_list
        }

    def regenerate_concept(self, brief: Dict[str, Any], original_concept: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        time.sleep(0.5)
        # Deep copy original and modify based on instruction
        new_concept = original_concept.copy()
        new_concept["title"] = f"{original_concept['title']} (Regenerated)"
        new_concept["main_visual_idea"] = f"{original_concept['main_visual_idea']} Modified for: {instruction}."
        new_concept["designer_execution_notes"] = f"{original_concept['designer_execution_notes']} Added focus: {instruction}."
        new_concept["reference_image_prompt"] = f"{original_concept['reference_image_prompt']}, modified with {instruction}"
        return new_concept

    def combine_concepts(self, brief: Dict[str, Any], concept_a: Dict[str, Any], concept_b: Dict[str, Any], custom_instruction: Optional[str] = None) -> Dict[str, Any]:
        time.sleep(0.8)
        combined_title = f"Hybrid: {concept_a['style_category']} + {concept_b['style_category']}"
        combined_prompt = f"Reference only: Hybrid image combining {concept_a['main_visual_idea']} and {concept_b['main_visual_idea']}"
        if custom_instruction:
            combined_prompt += f", modified by: {custom_instruction}"
            
        return {
            "concept_number": 99,  # Hybrid concept placeholder
            "title": combined_title,
            "style_category": "Hybrid Concept",
            "main_visual_idea": f"A fusion concept combining the structure of {concept_a['title']} with the mood of {concept_b['title']}.",
            "composition": f"Blended: {concept_a['composition']} combined with {concept_b['composition']}.",
            "lighting": f"Fused lighting: {concept_a['lighting']} / {concept_b['lighting']}.",
            "background": f"Merged backdrop: {concept_a['background']} mixed with {concept_b['background']}.",
            "color_palette": list(set(concept_a.get("color_palette", []) + concept_b.get("color_palette", []))),
            "typography_direction": f"Hybrid typography of {concept_a['typography_direction']} and {concept_b['typography_direction']}.",
            "creative_twist": "An unexpected intersection of styles.",
            "designer_execution_notes": "Pay close attention to blending transitions between both source concepts.",
            "reference_image_prompt": combined_prompt,
            "reference_only_notice": "Reference only - final artwork belongs to the designer."
        }

    def test_connection(self) -> bool:
        return True
