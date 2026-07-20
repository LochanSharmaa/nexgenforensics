# Prompts for SIFS Imagination Expander AI

BRIEF_EXTRACTION_PROMPT = """
Extract a structured creative brief from the user's raw imagination text.

User's Raw Imagination:
"{raw_imagination}"

Style Profile Context (if available):
{style_profile}

Return ONLY a JSON object matching this schema:
{{
  "main_subject": "core visual subject, e.g. perfume bottle floating above water",
  "design_type": "type of design, e.g. poster, digital ad, packaging",
  "target_audience": "e.g. luxury buyers, youth",
  "mood": ["mood word 1", "mood word 2"],
  "colors": ["color 1", "color 2"],
  "fixed_elements": ["element that must be kept"],
  "flexible_elements": ["element that can be modified"],
  "avoid_elements": ["element to avoid"]
}}
"""

QUESTIONS_GENERATION_PROMPT = """
You are given a structured creative brief. Generate 5 to 8 clarifying questions that will help narrow down the art direction, typography, lighting, and composition.

Creative Brief:
{brief}

Return a JSON array of strings containing ONLY the questions. Example format:
[
  "Should the product be centered or off-center?",
  "What type of texture should the background surface have?"
]
"""

CONCEPTS_GENERATION_PROMPT = """
Generate exactly 10 distinct visual concepts for the designer based on the creative brief and clarifying question answers.

Creative Brief:
{brief}

Clarifying Question Answers:
{answers}

Style Profile Preferences:
{style_profile}

The 10 concepts must correspond exactly to these 10 style categories (one concept per category):
1. Minimal luxury
2. Cinematic dramatic
3. Futuristic premium
4. Surreal dreamlike
5. Editorial magazine
6. Product advertisement
7. Abstract artistic
8. Dark premium
9. Human emotional story
10. Experimental poster

Self-Check Instruction:
Before finalizing, compare all 10 concepts. If any two feel too similar, rewrite one so each concept has a unique visual direction (different composition, lighting, background, metaphor, color accent, typography).

Every concept must have a "reference_image_prompt" prefixing "Reference only: " and must contain the "reference_only_notice" set to "Reference only - final artwork belongs to the designer."

Return ONLY a JSON object matching this schema:
{{
  "project_title": "string title",
  "brief_summary": "string brief summary",
  "concepts": [
    {{
      "concept_number": 1,
      "title": "string",
      "style_category": "Minimal luxury",
      "main_visual_idea": "detailed description of visual concept and metaphor",
      "composition": "detailed description of grid, frame, and structure",
      "lighting": "detailed description of light source, angle, intensity, shadow",
      "background": "detailed description of setting, backdrop, and texture",
      "color_palette": ["color 1", "color 2", "accent color"],
      "typography_direction": "detailed typography specification (font weight, style, spacing)",
      "creative_twist": "unique creative element or visual pun",
      "designer_execution_notes": "practical execution tips for the graphic designer",
      "reference_image_prompt": "Reference only: detailed visual prompt for an image generator",
      "reference_only_notice": "Reference only - final artwork belongs to the designer."
    }},
    ...
  ]
}}
"""

REGENERATE_CONCEPT_PROMPT = """
Regenerate this specific visual concept based on the feedback instruction.

Creative Brief:
{brief}

Original Concept:
{concept}

Feedback Instruction:
"{instruction}"

Return ONLY a JSON object matching this schema:
{{
  "concept_number": {concept_number},
  "title": "updated title",
  "style_category": "{style_category}",
  "main_visual_idea": "updated detailed visual concept",
  "composition": "updated composition details",
  "lighting": "updated lighting plan",
  "background": "updated background details",
  "color_palette": ["color 1", "color 2"],
  "typography_direction": "updated typography direction",
  "creative_twist": "updated creative twist",
  "designer_execution_notes": "updated designer notes",
  "reference_image_prompt": "Reference only: updated image prompt",
  "reference_only_notice": "Reference only - final artwork belongs to the designer."
}}
"""

COMBINE_CONCEPTS_PROMPT = """
Combine Concept A and Concept B into a new, single hybrid concept.

Creative Brief:
{brief}

Concept A:
{concept_a}

Concept B:
{concept_b}

Custom Instruction (if any):
"{instruction}"

Return ONLY a JSON object matching this schema:
{{
  "title": "hybrid title",
  "style_category": "Hybrid Concept",
  "main_visual_idea": "blended detailed visual concept",
  "composition": "blended composition",
  "lighting": "blended lighting",
  "background": "blended background",
  "color_palette": ["colors"],
  "typography_direction": "blended typography direction",
  "creative_twist": "unexpected blended twist",
  "designer_execution_notes": "execution tips for this blended concept",
  "reference_image_prompt": "Reference only: blended image prompt",
  "reference_only_notice": "Reference only - final artwork belongs to the designer."
}}
"""
