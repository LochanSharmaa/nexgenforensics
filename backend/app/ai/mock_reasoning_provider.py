import time
from typing import Dict, Any, List, Optional
from app.ai.base import LLMProvider

class MockReasoningProvider(LLMProvider):
    def extract_brief(self, raw_imagination: str, style_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        time.sleep(0.5)
        
        # Determine if perfume-focused or fallback
        is_perfume = "perfume" in raw_imagination.lower()
        
        subject = "A luxury perfume bottle floating above black water" if is_perfume else "Sleek premium product layout"
        design_type = "Luxury perfume poster" if is_perfume else "Conceptual brand poster"
        
        tensions = [
            "We identified a core tension between: 'minimal luxury' layout and the 'premium, narrative' mood of the floating bottle. Minimizing text elements conflicts with the rich, mystical storytelling implied by the floating metaphor.",
            "Tension between high-contrast chiaroscuro shadows (which hide product details) and luxury marketing needs (which require product clarity)."
        ] if is_perfume else [
            "Tension between minimalist space constraints and detailed storytelling.",
            "Tension between experimental layout risk and commercial audience clarity."
        ]

        return {
            "main_subject": subject,
            "design_type": design_type,
            "target_audience": "High-end fragrance consumers seeking poetry, mystery, and artistic design language" if is_perfume else "Design enthusiasts and premium buyers",
            "mood": ["dark", "mysterious", "premium", "luxury", "poetic"] if is_perfume else ["modern", "premium", "bold"],
            "colors": ["obsidian black", "deep navy", "silver", "moonbeam white"] if is_perfume else ["charcoal grey", "brushed gold", "matte black"],
            "fixed_elements": [subject, "minimalist typography layout"],
            "flexible_elements": ["angle of the light beams", "water ripple patterns", "cap texture reflection"],
            "avoid_elements": ["bright playful tones", "overstuffed environments", "generic commercial gloss textures"],
            "tensions": tensions
        }

    def generate_questions(self, brief: Dict[str, Any]) -> List[str]:
        time.sleep(0.4)
        return [
            "Should the typography be absolute dead-center (restrained) or dynamically cropped off-canvas (experimental)?",
            "Are the water reflections cold silver (high luxury) or warm gold (human emotional story)?",
            "Do we want to physically ground the bottle with a stone deck, or leave it floating as a surreal metaphor?",
            "Should we show the perfume brand name clearly, or let the bottle shape convey the brand entirely?",
            "What level of creative risk is acceptable? Safe ↔ Adventurous?"
        ]

    def generate_concepts(self, brief: Dict[str, Any], answers: List[Dict[str, Any]], style_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        time.sleep(1.0)
        
        subject = brief.get("main_subject", "A luxury perfume bottle")
        
        # Build 10 rich concepts representing the 10 selected lenses
        concepts_list = [
            {
                "concept_number": 1,
                "title": "Moonlit Reflection (The Control)",
                "style_category": "Literalist",
                "main_visual_idea": f"A literal execution of the brief: {subject} suspended 2 inches above calm black water, casting a silver reflection line.",
                "composition": "Centered product symmetry. Massive negative space framing the bottle. Clean serif type at the lower third.",
                "lighting": "Single overhead spotlight mimicking moonlight, casting soft silver rim reflections on the glass bottle.",
                "background": "Flat dark blue-black canvas fading into deep mist at the horizon line.",
                "color_palette": ["obsidian black", "deep navy", "moonlight silver"],
                "typography_direction": "Thin elegant serif typography, generous tracking, centered alignment.",
                "creative_twist": "The silver water reflection aligns exactly with the bottle's internal liquid level.",
                "designer_execution_notes": "Use absolute symmetry. Avoid props or ripples that disrupt the silent water reflection.",
                "reference_image_prompt": "Reference only: A centered luxury perfume bottle floating over still black water, single silver moonbeam, minimal elegant layout.",
                "reference_only_notice": "Reference only - final artwork belongs to the designer.",
                # Reasoning Chain
                "lens_applied": "Literalist",
                "tension_identified": "Standard brief alignment vs. visual stagnation.",
                "design_decision": "Adhere strictly to symmetry and quiet reflection vectors to set a base control concept.",
                "consequence": "Highly clean and recognizable design, but carries low surprise/disruption value.",
                "risk_flag": "Safe layout that might resemble generic luxury perfume ads.",
                # Metaphor
                "metaphor": "The bottle as a solitary lighthouse in the night.",
                "rejected_metaphor_1": "The bottle as a floating anchor.",
                "rejected_metaphor_2": "The bottle as a melting ice cube.",
                # Sub-panels
                "camera_language": "Eye-level direct shot, 85mm prime lens simulation, shallow depth of field.",
                "lighting_reasoning": "Standard moonbeam lighting to establish pure product form.",
                "material_reasoning": "High-gloss glass bottle finish, matte black paper print stock.",
                "anti_pattern": "Do not add excessive water splashes; keep the water glassy and undisturbed.",
                # Scores
                "novelty_score": 0.35,
                "feasibility_score": 0.95,
                "feasibility_reason": "Extremely straightforward studio photography layout.",
                "lens_fidelity_score": 0.98,
                "critic_notes": []
            },
            {
                "concept_number": 2,
                "title": "The Solar Eclipse Synthesis",
                "style_category": "Contradiction",
                "main_visual_idea": "A dark luxury poster set in a blinding white solar eclipse. Instead of moonlight reflection on black water, we see a black reflection on burning white water.",
                "composition": "Asymmetric, bottle placed on the right vertical grid line. Eclipse halo behind the cap.",
                "lighting": "Backlit corona casting a harsh silhouette, with high-contrast lens flares outlining the bottle's logo.",
                "background": "White-hot liquid water surface reflecting the sky, with the bottle casting an obsidian black shadow ripple.",
                "color_palette": ["solar gold", "corona white", "obsidian black"],
                "typography_direction": "Bold sans-serif typography, left-aligned, overlapping the bottle silhouette.",
                "creative_twist": "The shadow of the bottle forms the exact shape of a classic perfume atomizer bulb.",
                "designer_execution_notes": "Control the bloom of the corona light. The bottle must remain a legible silhouette.",
                "reference_image_prompt": "Reference only: Luxury perfume poster asymmetric composition, blinding solar eclipse backlight, gold white and black colors.",
                "reference_only_notice": "Reference only - final artwork belongs to the designer.",
                # Reasoning Chain
                "lens_applied": "Contradiction",
                "tension_identified": "Tension between 'dark mood' and 'moonlight reflection' - inverted to white sun reflection.",
                "design_decision": "Invert light and shadow values to make the reflection dark and the environment blindingly light.",
                "consequence": "Unusual high-contrast graphic appeal that defies classic night poster conventions.",
                "risk_flag": "Might feel too bright if colors are not controlled through high-contrast black silhouettes.",
                # Metaphor
                "metaphor": "The perfume bottle as a black hole sucking in the solar corona.",
                "rejected_metaphor_1": "A sun dial counting down to midnight.",
                "rejected_metaphor_2": "A floating charcoal block.",
                # Sub-panels
                "camera_language": "Low angle looking up, 24mm wide angle lens simulation, dramatic distortion.",
                "lighting_reasoning": "High contrast silhouette to create graphic visual weight.",
                "material_reasoning": "Frosted black glass bottle, metallic gold leaf accents, textured watercolor cardstock.",
                "anti_pattern": "Avoid mid-tone greys. Keep values strictly polar (stark black and bright gold).",
                "novelty_score": 0.88,
                "feasibility_score": 0.75,
                "feasibility_reason": "Silhouette lighting requires careful retouching to keep brand elements legible.",
                "lens_fidelity_score": 0.92,
                "critic_notes": []
            },
            {
                "concept_number": 3,
                "title": "The Scent of Absence",
                "style_category": "Subtraction",
                "main_visual_idea": "We remove the bottle entirely! The design shows only the floating water ripple and a silver moonlight reflection where the bottle should be.",
                "composition": "Floating center focus based on invisible geometry. Single silver reflection line warped into the shape of a bottle neck.",
                "lighting": "Diffused moonlight mapping a glass-like refractivity on the empty space.",
                "background": "Obsidian water ripples with microscopic glass fragments catching the starlight.",
                "color_palette": ["navy black", "liquid silver", "starlight white"],
                "typography_direction": "Ultralight sans-serif, widely letter-spaced, placed at the extreme margins.",
                "creative_twist": "The shape of the bottle is defined solely by how the surrounding water ripples warp around it.",
                "designer_execution_notes": "Use vector displacement mapping to create the refraction of an invisible bottle.",
                "reference_image_prompt": "Reference only: Invisible luxury perfume bottle shape refracting moonlight on calm black water.",
                "reference_only_notice": "Reference only - final artwork belongs to the designer.",
                # Reasoning Chain
                "lens_applied": "Subtraction",
                "tension_identified": "Tension between advertising a physical bottle and minimal luxury storytelling.",
                "design_decision": "Remove the physical product completely, using empty space and refraction to imply its shape.",
                "consequence": "Radical minimalism that forces the user to focus on the concept of scent and mystery.",
                "risk_flag": "Risk of client rejection since the physical product package is not shown.",
                # Metaphor
                "metaphor": "The product as a phantom or ghost scent.",
                "rejected_metaphor_1": "A sinking stone.",
                "rejected_metaphor_2": "A glass windowpane in the rain.",
                # Sub-panels
                "camera_language": "Macro close-up, 105mm lens, high focus falloff.",
                "lighting_reasoning": "Refractive starlight detailing the invisible volume.",
                "material_reasoning": "High-gloss metallic silver printing ink, pure velvet paper substrate.",
                "anti_pattern": "Do not let the bottle shape become too obscure; the refraction outline must be clear.",
                "novelty_score": 0.96,
                "feasibility_score": 0.50,
                "feasibility_reason": "Highly conceptual; requires rendering expertise rather than a simple photo.",
                "lens_fidelity_score": 0.97,
                "critic_notes": [
                    {
                        "agent_role": "Producer",
                        "note_text": "Removing the actual bottle from a perfume advertisement runs a massive risk of client rejection, as perfume buyers buy the bottle shape.",
                        "is_dissent": True
                    },
                    {
                        "agent_role": "Creative Rebel",
                        "note_text": "In luxury design, selling the concept of absence builds higher mystery and premium status. The bottle shape is implied, which is more memorable.",
                        "is_dissent": False
                    }
                ]
            },
            {
                "concept_number": 4,
                "title": "Nocturnal Chemistry",
                "style_category": "Wrong Medium",
                "main_visual_idea": "Designed as a 1970s Polish Film Poster. The bottle and water are hand-drawn in high-grain charcoal sketches and printed with offset silk-screened lettering.",
                "composition": "Loose, hand-composed structure. The bottle is distorted, merged with a surreal moon face.",
                "lighting": "Simulated harsh ink outline shadow, bold graphic ink washes.",
                "background": "Rough, yellowed newsprint paper texture with visible screen-print registration errors.",
                "color_palette": ["charcoal black", "bleach cream", "spot red label"],
                "typography_direction": "Heavy hand-drawn distressed serif fonts, offset aligned.",
                "creative_twist": "The moon's eye is positioned inside the glass cap of the perfume bottle.",
                "designer_execution_notes": "Incorporate ink splatters and misaligned silk-screen color blocks to mimic vintage print processes.",
                "reference_image_prompt": "Reference only: Vintage Polish film poster design, charcoal sketch, surreal perfume bottle illustration, offset inks.",
                "reference_only_notice": "Reference only - final artwork belongs to the designer.",
                # Reasoning Chain
                "lens_applied": "Wrong Medium",
                "tension_identified": "Digital product mockups vs. artistic hand-crafted authenticity.",
                "design_decision": "Transplant a perfume advertisement into the medium of vintage poster printmaking.",
                "consequence": "Radical hand-hewn feel that immediately separates it from sterile 3D renders.",
                "risk_flag": "Might look too retro or dusty for a modern luxury launch.",
                # Metaphor
                "metaphor": "The bottle as a charcoal sketch from a dream diary.",
                "rejected_metaphor_1": "A computer-generated wireframe.",
                "rejected_metaphor_2": "A photograph printed on glossy foil.",
                # Sub-panels
                "camera_language": "Flat scan representation, zero depth of field simulation.",
                "lighting_reasoning": "Graphic ink contrast rather than realistic volumetric light.",
                "material_reasoning": "Heavyweight recycled cotton pulp paper with rough deckle edges.",
                "anti_pattern": "Do not clean up the edges. Let the charcoal dust smear and register errors show.",
                "novelty_score": 0.85,
                "feasibility_score": 0.70,
                "feasibility_reason": "Requires high illustrative skill or custom vector brush styling.",
                "lens_fidelity_score": 0.90,
                "critic_notes": []
            },
            {
                "concept_number": 5,
                "title": "Industrial Oxidation",
                "style_category": "Audience Inversion",
                "main_visual_idea": "Designed specifically to alienate mainstream luxury buyers. The perfume bottle is reframed as a heavy, rusted iron industrial relic floating in oil-slicked muddy water.",
                "composition": "Brutalist center block, thick heavy borders, technical engineering labels.",
                "lighting": "Harsh, un-retouched industrial fluorescent tube lighting casting green-yellow tint spill.",
                "background": "Corrugated iron sheet wall, muddy brown water with toxic oil-slick iridescence.",
                "color_palette": ["rust orange", "toxic teal", "diesel brown"],
                "typography_direction": "Monospaced technical terminal font, right-justified.",
                "creative_twist": "The bottle cap is replaced by an iron gas pipe valve wheel.",
                "designer_execution_notes": "Add realistic rust textures and oil displacement ripples. Frame with high-contrast grunge filters.",
                "reference_image_prompt": "Reference only: Heavy rusted iron perfume bottle in toxic muddy water, fluorescent lights, brutalist aesthetic.",
                "reference_only_notice": "Reference only - final artwork belongs to the designer.",
                # Reasoning Chain
                "lens_applied": "Audience Inversion",
                "tension_identified": "Polished luxury codes vs. sub-cultural industrial grunge.",
                "design_decision": "Invert target aesthetic, replacing cleanliness and gold with rust, oil, and Monospace labels.",
                "consequence": "Highly polarizing visual that appeals strongly to avant-garde subcultures (brutalist/grunge).",
                "risk_flag": "Will alienate 95% of typical perfume buyers; target must be extremely niche.",
                # Metaphor
                "metaphor": "Scent as an industrial chemical compound.",
                "rejected_metaphor_1": "A delicate flower in a field.",
                "rejected_metaphor_2": "A crystal goblet.",
                # Sub-panels
                "camera_language": "Direct flash, flat perspective, high-grain digital sensor look.",
                "lighting_reasoning": "Fluorescent raw lighting to strips away 'studio luxury' pretensions.",
                "material_reasoning": "Raw oxidized steel, unpolished iron plates, recycled corrugated cardboard.",
                "anti_pattern": "Do not soften the rust textures; keep the corrosion sharp and aggressive.",
                "novelty_score": 0.92,
                "feasibility_score": 0.80,
                "feasibility_reason": "Easy to photograph if mock props are sourced, but hard to sell to standard clients.",
                "lens_fidelity_score": 0.94,
                "critic_notes": []
            },
            {
                "concept_number": 6,
                "title": "Metallic Embossing",
                "style_category": "Material Honesty",
                "main_visual_idea": "The layout is generated entirely by the physical constraints of hot-foil stamping. The bottle outline and water lines are pressed directly as silver foil lines into a thick, textured black cotton stock.",
                "composition": "Minimalist contours. The bottle cap is a solid foil square, and the water is represented by three silver curved lines.",
                "lighting": "Simulated angled light reflecting off the metallic foil deboss edges.",
                "background": "Deeply textured black letterpress paper stock with natural cotton fibers.",
                "color_palette": ["matte carbon black", "hot-stamp silver foil"],
                "typography_direction": "Traditional block serif typography, debossed deeply without ink.",
                "creative_twist": "The text is completely invisible unless light hits the debossed paper at a 45-degree angle.",
                "designer_execution_notes": "Use vector path outlines only. Ensure no gradients or mid-tones are used, as letterpress foil cannot reproduce them.",
                "reference_image_prompt": "Reference only: Silver foil stamp debossed on thick black textured cotton paper, minimalist perfume outline.",
                "reference_only_notice": "Reference only - final artwork belongs to the designer.",
                # Reasoning Chain
                "lens_applied": "Material Honesty",
                "tension_identified": "Digital screen rendering vs. physical print-finishing constraints.",
                "design_decision": "Limit layout to vectors that can be manufactured via custom metal embossing dies.",
                "consequence": "High sensory physical appeal; clean minimal luxury that values the tactile print stock.",
                "risk_flag": "Cannot be rendered effectively on low-contrast digital screens; requires premium printing budget.",
                # Metaphor
                "metaphor": "The design as a tactile stamp or coin.",
                "rejected_metaphor_1": "A digital painting.",
                "rejected_metaphor_2": "A floating vapor.",
                # Sub-panels
                "camera_language": "Angled macro photograph showing depth of deboss, 50mm lens.",
                "lighting_reasoning": "Single hard raking light source to illuminate foil edges.",
                "material_reasoning": "600gsm cotton letterpress board, metallic hot-stamping foil.",
                "anti_pattern": "Do not include print gradients. The foil must be 100% solid or 100% absent.",
                "novelty_score": 0.74,
                "feasibility_score": 0.88,
                "feasibility_reason": "Straightforward production if letterpress vector constraints are kept.",
                "lens_fidelity_score": 0.96,
                "critic_notes": []
            },
            {
                "concept_number": 7,
                "title": "The Shattered Mirage",
                "style_category": "Narrative Compression",
                "main_visual_idea": "The bottle is half-merged with the black water, dissolving into liquid silver, capturing the exact transition of physical state. It is frame 1 of a scent dissolving.",
                "composition": "Diagonal split. The top-left is sharp floating glass; the bottom-right is melting silver liquid.",
                "lighting": "Strobe-flashed water splashes freeze-framing the silver droplets in mid-air.",
                "background": "Violent dark ripples and water vortex paths reflecting navy and violet sky tones.",
                "color_palette": ["liquid silver", "obsidian navy", "violet stardust"],
                "typography_direction": "Italicized serif, right-aligned, conveying movement and chemical reaction.",
                "creative_twist": "The melting silver droplets form the letters of the perfume's name.",
                "designer_execution_notes": "Use high-speed liquid simulations to merge the glass geometry with silver fluid paths.",
                "reference_image_prompt": "Reference only: Glass perfume bottle dissolving into liquid silver over turbulent black water, macro flash photography.",
                "reference_only_notice": "Reference only - final artwork belongs to the designer.",
                # Reasoning Chain
                "lens_applied": "Narrative Compression",
                "tension_identified": "Static product placement vs. dynamic temporal storytelling.",
                "design_decision": "Freeze the frame at the threshold of the bottle melting into the water.",
                "consequence": "Dramatic high-action visual that conveys perfume application as a chemical transition.",
                "risk_flag": "Requires sophisticated CGI/fluid simulation to execute realistically.",
                # Metaphor
                "metaphor": "Scent as a melting ice sculpture.",
                "rejected_metaphor_1": "A static stone pedestal.",
                "rejected_metaphor_2": "A vapor trail in the sky.",
                # Sub-panels
                "camera_language": "High-speed macro action capture, 1/8000s shutter simulation.",
                "lighting_reasoning": "High-speed strobe lighting to freeze fluid motion highlights.",
                "material_reasoning": "Foil laminate packaging box, premium synthetic print paper.",
                "anti_pattern": "Do not let the melting transition look like low-quality mud; it must read as premium liquid silver.",
                "novelty_score": 0.82,
                "feasibility_score": 0.60,
                "feasibility_reason": "CGI liquid morph rendering is complex to draft without specialized artists.",
                "lens_fidelity_score": 0.89,
                "critic_notes": []
            },
            {
                "concept_number": 8,
                "title": "Obsidian Desolation",
                "style_category": "Emotional Extremity",
                "main_visual_idea": "A concept designed to evoke feelings of absolute isolation and cosmic desolation. A tiny, single perfume bottle floats in a vast, empty, pitch-black ocean under a starless sky.",
                "composition": "Extreme negative space. The bottle occupies less than 2% of the canvas area, placed in the far lower corner.",
                "lighting": "A tiny, pin-prick ray of cold white light hitting only the cap of the bottle, leaving the main body in darkness.",
                "background": "Vast endless black water ripples extending to a completely blank, dark sky. Zero stars or moon.",
                "color_palette": ["stark black", "cold coal grey", "pinpoint white"],
                "typography_direction": "Microscopic sans-serif, positioned at the center axis, separated by large spacing.",
                "creative_twist": "The branding text is so small that it forces the viewer to lean in and inspect the poster closely.",
                "designer_execution_notes": "Maximize black levels. Ensure the printer can handle deep obsidian values without crushing the highlight detail.",
                "reference_image_prompt": "Reference only: Tiny luxury perfume bottle in a vast empty pitch black ocean, high isolation, dark minimalism.",
                "reference_only_notice": "Reference only - final artwork belongs to the designer.",
                # Reasoning Chain
                "lens_applied": "Emotional Extremity",
                "tension_identified": "Standard commercial size ratios vs. emotional desolation.",
                "design_decision": "Shrink the product size to a microscopic scale and hide starlight to amplify isolation.",
                "consequence": "Striking, solemn, and moody poster that demands attention through silent space.",
                "risk_flag": "Violates the basic advertising rule of keeping the product large and legible.",
                # Metaphor
                "metaphor": "The product as a distant, dying star.",
                "rejected_metaphor_1": "A loud billboard.",
                "rejected_metaphor_2": "A nested jewel box.",
                # Sub-panels
                "camera_language": "Extremely far telephoto shot, bird's-eye perspective looking straight down.",
                "lighting_reasoning": "Minimal light key to evoke isolation and premium quietude.",
                "material_reasoning": "Matte velvet charcoal poster board, zero reflection varnish.",
                "anti_pattern": "Do not add stars, clouds, or boats. The canvas must remain absolute silent black.",
                "novelty_score": 0.80,
                "feasibility_score": 0.90,
                "feasibility_reason": "Requires simple layout scaling but high quality black level control.",
                "lens_fidelity_score": 0.91,
                "critic_notes": []
            },
            {
                "concept_number": 9,
                "title": "Brutalist Concrete Scent",
                "style_category": "Cultural Counter-Signal",
                "main_visual_idea": "Refuses standard luxury tropes. The bottle is positioned on a cracked grey concrete slab. No moonlight, no water. Flat neon orange labeling sits directly on the glass.",
                "composition": "Off-center brutalist grid, heavy vertical borders, zero elegance.",
                "lighting": "Harsh, direct studio flash from camera axis, casting flat dark shadows behind the bottle.",
                "background": "Industrial grey concrete wall with structural water stain streaks.",
                "color_palette": ["concrete grey", "asphalt black", "safety neon orange"],
                "typography_direction": "Bold grotesque sans-serif font, squashed tracking, massive scale.",
                "creative_twist": "The bottle looks like a medical fluid container rather than an expensive perfume flask.",
                "designer_execution_notes": "Use flat direct shadows, avoid rim lighting or glass glow, emphasize concrete textures.",
                "reference_image_prompt": "Reference only: Brutalist perfume advertisement, concrete wall, direct flash photography, neon labels.",
                "reference_only_notice": "Reference only - final artwork belongs to the designer.",
                # Reasoning Chain
                "lens_applied": "Cultural Counter-Signal",
                "tension_identified": "Traditional luxurious code vs. industrial brutalist counter-culture.",
                "design_decision": "Abandon luxury moonlit water for a raw concrete slab and a flat camera flash.",
                "consequence": "Highly edgy and modern visual that commands attention by refusing commercial beauty standards.",
                "risk_flag": "May look cheap to customers who do not understand the brutalist aesthetic.",
                # Metaphor
                "metaphor": "Fragrance as an urban construct or structural concrete form.",
                "rejected_metaphor_1": "A royal glass crown.",
                "rejected_metaphor_2": "A drop of morning dew.",
                # Sub-panels
                "camera_language": "Direct frontal eye-level view, 50mm lens simulation, flat perspective.",
                "lighting_reasoning": "Direct camera flash to replicate raw streetwear lookbooks.",
                "material_reasoning": "Uncoated matte cardstock, heavy cardboard texture.",
                "anti_pattern": "Do not soften the concrete cracks; keep the textures raw and unpolished.",
                "novelty_score": 0.89,
                "feasibility_score": 0.95,
                "feasibility_reason": "Very easy to shoot with simple equipment.",
                "lens_fidelity_score": 0.93,
                "critic_notes": []
            },
            {
                "concept_number": 10,
                "title": "The Single Line",
                "style_category": "Restraint Maximalism",
                "main_visual_idea": "The layout is reduced to its absolute limit: a single, horizontal, 1px silver vector line on a solid black canvas, representing the bottle's reflection.",
                "composition": "One single horizontal line splitting the canvas, with microscopic brand initials at the center.",
                "lighting": "None. Flat digital colors only.",
                "background": "Solid absolute digital black (#000000).",
                "color_palette": ["solid black", "silver line"],
                "typography_direction": "Microscopic thin sans-serif, letter-spaced, placed directly on the line.",
                "creative_twist": "The line is broken for exactly 3 pixels to reveal the silhouette of the bottle's base.",
                "designer_execution_notes": "Maintain maximum grid control. Ensure the line is perfectly aligned to the pixel grid.",
                "reference_image_prompt": "Reference only: Minimalist black poster with a single silver horizontal vector line, luxury branding concept.",
                "reference_only_notice": "Reference only - final artwork belongs to the designer.",
                # Reasoning Chain
                "lens_applied": "Restraint Maximalism",
                "tension_identified": "Textual information vs. extreme visual restraint.",
                "design_decision": "Reduce the entire visual composition to a single pixel line and minor typography.",
                "consequence": "Ultra-premium, mysterious layout that borders on fine art print.",
                "risk_flag": "Extremely abstract; does not communicate product details to typical consumers.",
                # Metaphor
                "metaphor": "The design as a single string on an instrument.",
                "rejected_metaphor_1": "A dense dictionary page.",
                "rejected_metaphor_2": "A starry night sky.",
                # Sub-panels
                "camera_language": "Vector graphic rendering, zero camera simulation.",
                "lighting_reasoning": "Flat graphic values with no lighting layers.",
                "material_reasoning": "High-density black acrylic board with laser-etched silver inlay.",
                "anti_pattern": "Do not add glowing gradients to the silver line; it must remain a flat solid vector.",
                "novelty_score": 0.85,
                "feasibility_score": 0.98,
                "feasibility_reason": "Easiest layout to execute digitally.",
                "lens_fidelity_score": 0.95,
                "critic_notes": []
            }
        ]
        
        return {
            "project_title": brief.get("main_subject", "Expanded Project"),
            "brief_summary": f"Visual concepts generated from: '{brief.get('main_subject')}' brief.",
            "concepts": concepts_list
        }

    def regenerate_concept(self, brief: Dict[str, Any], original_concept: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        time.sleep(0.5)
        new_concept = original_concept.copy()
        new_concept["title"] = f"{original_concept['title']} (Regenerated V2)"
        new_concept["main_visual_idea"] = f"{original_concept['main_visual_idea']} Updated with instruction: {instruction}"
        new_concept["designer_execution_notes"] = f"{original_concept['designer_execution_notes']} Added focus: {instruction}"
        
        # Mutation updates in reasoning chain
        new_concept["design_decision"] = f"{original_concept['design_decision']} Modifying logic: {instruction}"
        new_concept["consequence"] = f"{original_concept['consequence']} (Mutated under instruction: {instruction})"
        
        return new_concept

    def combine_concepts(self, brief: Dict[str, Any], concept_a: Dict[str, Any], concept_b: Dict[str, Any], custom_instruction: Optional[str] = None) -> Dict[str, Any]:
        time.sleep(0.6)
        combined_title = f"Hybrid: {concept_a['style_category']} + {concept_b['style_category']}"
        
        return {
            "concept_number": 99,
            "title": combined_title,
            "style_category": "Hybrid Concept",
            "main_visual_idea": f"A hybrid reasoning layout combining A ({concept_a['title']}) and B ({concept_b['title']}).",
            "composition": f"Blended grid: {concept_a['composition']} + {concept_b['composition']}",
            "lighting": f"Combined lighting: {concept_a['lighting']} + {concept_b['lighting']}",
            "background": f"Merged background: {concept_a['background']} + {concept_b['background']}",
            "color_palette": list(set(concept_a.get("color_palette", []) + concept_b.get("color_palette", []))),
            "typography_direction": f"Hybrid font pairings of {concept_a['typography_direction']} and {concept_b['typography_direction']}",
            "creative_twist": "Unexpected fusion of conceptual directions.",
            "designer_execution_notes": "Balance the extreme aspects of both source visual strategies.",
            "reference_image_prompt": f"Reference only: Blended visual layout combining {concept_a['style_category']} and {concept_b['style_category']}",
            "reference_only_notice": "Reference only - final artwork belongs to the designer.",
            # Reasoning Chain
            "lens_applied": "Hybrid Concept",
            "tension_identified": f"Merging the visual philosophy of {concept_a['style_category']} and {concept_b['style_category']}.",
            "design_decision": "Fuse the structural composition of Concept A with the materials of Concept B.",
            "consequence": "Creates a novel category-blurring design option.",
            "risk_flag": "Risk of visual confusion if execution is too cluttered.",
            # Metaphor
            "metaphor": "A double-exposure creative print.",
            "rejected_metaphor_1": "Flat single canvas.",
            "rejected_metaphor_2": "Normal studio render.",
            # Sub-panels
            "camera_language": concept_a.get("camera_language", "50mm camera lens"),
            "lighting_reasoning": concept_a.get("lighting_reasoning", "Studio ambient lighting"),
            "material_reasoning": concept_b.get("material_reasoning", "Premium printing paper"),
            "anti_pattern": "Do not let elements fight for hierarchy. Choose one clear anchor.",
            # Scores
            "novelty_score": 0.85,
            "feasibility_score": 0.70,
            "feasibility_reason": "Fusing two styles requires active layout decisions.",
            "lens_fidelity_score": 0.90,
            "critic_notes": []
        }

    def test_connection(self) -> bool:
        return True
