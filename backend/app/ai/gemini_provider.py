import os
import json
import logging
import asyncio
import time
import hashlib
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from pydantic import BaseModel, Field
from sqlmodel import Session, select
import redis

from app.core.config import settings
from app.ai.base import LLMProvider
from app.db.database import engine
from app.models.reasoning import Lens
from app.models.additional_models import ProviderLog

logger = logging.getLogger(__name__)

# Initialize Redis client
try:
    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
except Exception as rx:
    logger.warning(f"GeminiProvider: Failed to connect to Redis, caching disabled: {str(rx)}")
    redis_client = None

# Schemas for Gemini Structured JSON Outputs
class BriefExtraction(BaseModel):
    main_subject: str = Field(description="The core subject of the design.")
    design_type: str = Field(description="The format or category of design (e.g. poster, packaging).")
    target_audience: str = Field(description="The primary target consumer/demographic.")
    mood: List[str] = Field(description="List of 3-5 visual/mood keywords.")
    colors: List[str] = Field(description="List of 3-5 color palette guidelines.")
    fixed_elements: List[str] = Field(description="Elements that must remain unchanged.")
    flexible_elements: List[str] = Field(description="Elements where the designer has creative freedom.")
    avoid_elements: List[str] = Field(description="Aesthetics, subjects, or tropes to avoid.")
    tensions: List[str] = Field(description="Explicit internal tensions or contradictions identified in the brief.")

class LensSelectionItem(BaseModel):
    lens_id: int
    lens_name: str
    justification: str

class LensSelectionResponse(BaseModel):
    selected_lenses: List[LensSelectionItem]

class MetaphorRejectedItem(BaseModel):
    metaphor: str
    reason: str

class ConceptGenerationResponse(BaseModel):
    title: str = Field(description="Title of the concept.")
    main_visual_idea: str = Field(description="The core visual idea of the design.")
    composition: str = Field(description="The layout, hierarchy, and geometric composition.")
    lighting: str = Field(description="The lighting design and mood.")
    background: str = Field(description="The background or environmental details.")
    color_palette: List[str] = Field(description="Detailed color palette (3-5 specific colors).")
    typography_direction: str = Field(description="Typography choices and layout instructions.")
    creative_twist: str = Field(description="An unexpected creative spark or twist.")
    designer_execution_notes: str = Field(description="Technical advice and implementation tips for the human designer.")
    reference_image_prompt: str = Field(description="Detailed prompt for generating a reference image (should begin with 'Reference only: ').")
    
    # Reasoning Chain fields
    lens_applied: str = Field(description="The name of the lens applied.")
    tension_identified: str = Field(description="The specific brief tension this concept addresses.")
    design_decision: str = Field(description="The key design decision that resolves this tension.")
    consequence: str = Field(description="The visual or cognitive consequence of this decision.")
    risk_flag: str = Field(description="The creative or execution risk associated with this decision.")
    
    # Metaphors
    metaphor_primary: str = Field(description="The primary visual metaphor of the design.")
    metaphor_rejected: List[MetaphorRejectedItem] = Field(description="Exactly two rejected metaphors with one-sentence explanation of why they were rejected.")
    
    # Technical sub-panels
    camera_language: str = Field(description="Camera lens, angle, depth of field description.")
    lighting_reasoning: str = Field(description="Detailed logic for the lighting scheme chosen.")
    material_reasoning: str = Field(description="Physical material, ink, or print substrate choice.")
    anti_pattern: str = Field(description="A line describing what a less-thoughtful execution of this idea would get wrong.")

class ConceptCritiqueItem(BaseModel):
    concept_number: int
    title: str
    feasibility_score: int = Field(description="Feasibility score from 1-5.")
    feasibility_reason: str = Field(description="One-line justification for feasibility score.")
    trend_similarity_flag: bool = Field(description="True if it resembles an oversaturated trend or other concept in the set.")
    trend_similarity_notes: Optional[str] = Field(None, description="Notes on what trend or concept it resembles.")

class DissentNoteItem(BaseModel):
    concept_number: int
    note: str
    is_dissent: bool = Field(default=True)

class CriticPanelResponse(BaseModel):
    per_concept: List[ConceptCritiqueItem]
    cross_concept_critique: str = Field(description="One paragraph of overall cross-concept critique.")
    dissent_notes: List[DissentNoteItem] = Field(description="Dissenting opinions from the critic roles.")

class RewriteInstructionItem(BaseModel):
    concept_number: int
    instruction: str

class CreativeDirectorRewriteResponse(BaseModel):
    rewrites: List[RewriteInstructionItem]

class MissingOpportunityResponse(BaseModel):
    report_text: str = Field(description="A comprehensive analysis of missing design directions or opportunities based on the brief and generated concepts.")

class GeminiProvider(LLMProvider):
    def __init__(self):
        api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            genai.configure(api_key=api_key)
        self.pro_model_name = settings.GEMINI_PRO_MODEL
        self.flash_model_name = settings.GEMINI_FLASH_MODEL

    def _log_provider_call(self, task_type: str, status: str, latency_ms: int, error_msg: Optional[str] = None):
        """Log latency and status of a Gemini API call into provider_logs table."""
        try:
            with Session(engine) as session:
                log_entry = ProviderLog(
                    provider="gemini",
                    task_type=task_type,
                    status=status,
                    latency_ms=latency_ms,
                    error_message_sanitized=error_msg[:1000] if error_msg else None
                )
                session.add(log_entry)
                session.commit()
        except Exception as ex:
            logger.error(f"GeminiProvider: Failed to log call history: {ex}")

    def _call_gemini_structured(self, model_name: str, prompt: str, schema: BaseModel, system_instruction: Optional[str] = None, task_type: str = "llm_call") -> Any:
        """Call Gemini model and return validated structured data matching the schema. Retries once on fail."""
        start_time = time.time()
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction
            )
            config = GenerationConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.7
            )
            response = model.generate_content(prompt, generation_config=config)
            validated = schema.model_validate_json(response.text)
            
            latency = int((time.time() - start_time) * 1000)
            self._log_provider_call(task_type, "success", latency)
            return validated
        except Exception as e:
            # Retry once with the error message in the prompt
            logger.warning(f"First attempt failed for structured call: {e}. Retrying once...")
            try:
                retry_prompt = f"{prompt}\n\n[System Alert: The previous response failed Pydantic validation with error: {str(e)}. Please correct your output format to exactly match the JSON schema.]"
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instruction
                )
                config = GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.4 # lower temp to ensure structure correctness
                )
                response = model.generate_content(retry_prompt, generation_config=config)
                validated = schema.model_validate_json(response.text)
                
                latency = int((time.time() - start_time) * 1000)
                self._log_provider_call(task_type, "success", latency)
                return validated
            except Exception as retry_err:
                latency = int((time.time() - start_time) * 1000)
                self._log_provider_call(task_type, "failure", latency, error_msg=str(retry_err))
                logger.error(f"Gemini API structured call retry failed: {str(retry_err)}")
                raise retry_err

    def extract_brief(self, raw_imagination: str, style_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        global_identity = (
            "You are part of SIFS Creative Reasoning Engine, a multi-agent creative-direction system for "
            "professional graphic designers. Your outputs are never final artwork. You produce reasoning "
            "and reference-only visual direction. The designer is always the final author. Never claim "
            "certainty about what is 'good' — show your reasoning instead. Return clean structured JSON "
            "matching the provided schema exactly. No markdown fences, no preamble."
        )
        user_prompt = (
            f"Read this raw imagination/brief. Extract explicit internal tensions between stated elements "
            f"(conflicts AND reinforcements — e.g., 'minimal' vs 'emotional story' conflict; 'premium' and "
            f"'minimal text' reinforce each other). Match the specificity of a senior creative director in "
            f"the first five minutes of a briefing — no generic tension-naming.\n\n"
            f"Raw Imagination:\n{raw_imagination}"
        )
        if style_profile:
            user_prompt += f"\n\nStyle Profile Preferences:\n{json.dumps(style_profile)}"
            
        res = self._call_gemini_structured(
            model_name=self.flash_model_name,
            prompt=user_prompt,
            schema=BriefExtraction,
            system_instruction=global_identity,
            task_type="extract_brief"
        )
        return res.model_dump()

    def generate_questions(self, brief: Dict[str, Any]) -> List[str]:
        start_time = time.time()
        try:
            model = genai.GenerativeModel(
                model_name=self.flash_model_name,
                system_instruction="You are SIFS design researcher. Generate exactly 5-8 highly specific clarifying questions."
            )
            prompt = (
                f"Review the extracted creative brief:\n{json.dumps(brief)}\n\n"
                f"Produce exactly 5-8 sharp, professional clarifying questions for the designer that will resolve "
                f"ambiguity or explore design opportunities."
            )
            response = model.generate_content(prompt)
            latency = int((time.time() - start_time) * 1000)
            self._log_provider_call("generate_questions", "success", latency)
            
            questions = []
            for line in response.text.split("\n"):
                line = line.strip().lstrip("0123456789.-* ")
                if line:
                    questions.append(line)
            return questions[:8]
        except Exception as e:
            latency = int((time.time() - start_time) * 1000)
            self._log_provider_call("generate_questions", "failure", latency, error_msg=str(e))
            raise e

    def generate_concepts(self, brief: Dict[str, Any], answers: List[Dict[str, Any]], style_profile: Optional[Dict[str, Any]] = None, creative_distance: int = 3) -> Dict[str, Any]:
        """Runs the entire 6-pass agentic pipeline using asyncio."""
        return asyncio.run(self._generate_concepts_async(brief, answers, style_profile, creative_distance))

    async def _generate_concepts_async(self, brief: Dict[str, Any], answers: List[Dict[str, Any]], style_profile: Optional[Dict[str, Any]] = None, creative_distance: int = 3) -> Dict[str, Any]:
        # 1. Fetch available Lenses from database
        try:
            with Session(engine) as session:
                db_lenses = session.exec(select(Lens).where(Lens.active == True)).all()
        except Exception:
            db_lenses = []
            
        if not db_lenses:
            # Fallback to seed list if DB is empty
            db_lenses = [
                Lens(id=1, name="Literalist", description="Executes the brief exactly as stated.", reasoning_move="Pure control.", risk_level=1),
                Lens(id=2, name="Contradiction", description="Finds tension and makes it the subject.", reasoning_move="Dialectical synthesis.", risk_level=4),
                Lens(id=3, name="Subtraction", description="Removes the key obvious element.", reasoning_move="Obvious element deletion.", risk_level=3)
            ]

        # Calculate a brief hash for Redis caching of individual concepts
        brief_serialized = json.dumps({"brief": brief, "answers": answers, "style_profile": style_profile}, sort_keys=True)
        brief_hash = hashlib.md5(brief_serialized.encode()).hexdigest()

        # 2. Creative Director Pass 1 (Lens Selection with python-validation loop)
        lenses_list_str = "\n".join([f"- ID {l.id}: {l.name} | {l.description} | Move: {l.reasoning_move} | Risk: {l.risk_level}" for l in db_lenses])
        
        # Adjust lens selection style based on creative distance
        risk_guideline = "Balanced risk profile."
        if creative_distance <= 2:
            risk_guideline = "Weight heavily toward lower-risk, highly grounded, functional lenses (risk_level <= 3)."
        elif creative_distance >= 4:
            risk_guideline = "Weight heavily toward high-risk, unconventional, speculative lenses (risk_level >= 4)."

        cd_instruction = (
            "You are the Creative Director. Given this brief and its extracted tensions, select exactly "
            "10 lenses from the provided library. Requirements:\n"
            "1. Always include 'Literalist' (ID 1) as anchor;\n"
            "2. Always include at least one lens with risk_level >= 4;\n"
            f"3. {risk_guideline}\n"
            "4. Maximize pairwise distinctness of REASONING TYPE across the 10 for THIS brief specifically — "
            "a generic justification that could apply to any brief is a failure.\n"
            "Return JSON: { \"selected_lenses\": [{ \"lens_id\", \"lens_name\", \"justification\" }] }"
        )
        
        brief_data = f"Subject: {brief.get('main_subject')}\nType: {brief.get('design_type')}\nTensions: {brief.get('tensions')}"
        
        # Validation Loop
        max_attempts = 3
        cd_selection = None
        for attempt in range(max_attempts):
            try:
                cd_selection = self._call_gemini_structured(
                    model_name=self.pro_model_name,
                    prompt=f"Brief:\n{brief_data}\n\nAvailable Lenses:\n{lenses_list_str}",
                    schema=LensSelectionResponse,
                    system_instruction=cd_instruction,
                    task_type="lens_selection"
                )
                
                selected_ids = [item.lens_id for item in cd_selection.selected_lenses]
                
                # Check constraints
                literalist_lens = next((l for l in db_lenses if l.name == "Literalist"), None)
                has_literalist = literalist_lens and literalist_lens.id in selected_ids
                
                has_high_risk = False
                for item in cd_selection.selected_lenses:
                    lens_obj = next((l for l in db_lenses if l.id == item.lens_id), None)
                    if lens_obj and lens_obj.risk_level >= 4:
                        has_high_risk = True
                        break
                
                is_distinct = len(set(selected_ids)) == len(selected_ids)
                is_exactly_10 = len(cd_selection.selected_lenses) == 10

                if has_literalist and has_high_risk and is_distinct and is_exactly_10:
                    logger.info(f"CD lens selection validated successfully on attempt {attempt + 1}")
                    break
                else:
                    logger.warning(f"CD selection failed constraints on attempt {attempt + 1}. Literalist: {has_literalist}, HighRisk: {has_high_risk}, Distinct: {is_distinct}, Count: {len(cd_selection.selected_lenses)}")
            except Exception as ex:
                logger.error(f"CD selection failed: {ex}")
                if attempt == max_attempts - 1:
                    raise ex

        # Fallback enforcement if validation loop failed
        if not cd_selection or len(cd_selection.selected_lenses) < 10:
            logger.warning("CD selection failed validation constraints. Forcing fallback lens selection.")
            fallback_items = []
            # Literal list first
            lit = next((l for l in db_lenses if l.name == "Literalist"), db_lenses[0])
            fallback_items.append(LensSelectionItem(lens_id=lit.id, lens_name=lit.name, justification="Anchor Literalist control."))
            
            # Find a risk level >= 4 lens
            high_risk = next((l for l in db_lenses if l.risk_level >= 4), db_lenses[1])
            fallback_items.append(LensSelectionItem(lens_id=high_risk.id, lens_name=high_risk.name, justification="Required high-risk direction."))
            
            # Fill the rest with unique lenses
            seen_ids = {lit.id, high_risk.id}
            for l in db_lenses:
                if len(fallback_items) >= 10:
                    break
                if l.id not in seen_ids:
                    fallback_items.append(LensSelectionItem(lens_id=l.id, lens_name=l.name, justification=f"Divergent lens {l.name}."))
                    seen_ids.add(l.id)
            cd_selection = LensSelectionResponse(selected_lenses=fallback_items)

        # 3. Concept Generation (Parallel dispatch with Semaphore + Redis caching)
        semaphore = asyncio.Semaphore(settings.GEMINI_CONCURRENT_LIMIT)
        
        async def generate_single_concept(lens_item: LensSelectionItem, idx: int):
            async with semaphore:
                # Cache lookup
                cache_key = f"concept_cache:{brief_hash}:{lens_item.lens_id}"
                if redis_client:
                    try:
                        cached = redis_client.get(cache_key)
                        if cached:
                            logger.info(f"Redis Cache Hit for concept brief={brief_hash} lens={lens_item.lens_id}")
                            c_cached_data = json.loads(cached)
                            c_cached_data["concept_number"] = idx
                            return {"chat": None, "data": c_cached_data, "lens_id": lens_item.lens_id}
                    except Exception as cx:
                        logger.warning(f"Concept cache read failed: {cx}")

                lens_obj = next((l for l in db_lenses if l.id == lens_item.lens_id), None)
                lens_name = lens_obj.name if lens_obj else lens_item.lens_name
                lens_move = lens_obj.reasoning_move if lens_obj else "Custom Move"
                
                sys_prompt = (
                    "You are part of SIFS Creative Reasoning Engine, a multi-agent creative-direction system for "
                    "professional graphic designers. Your outputs are never final artwork. You produce reasoning "
                    "and reference-only visual direction. Never claim certainty about what is 'good' — show your reasoning instead.\n\n"
                    f"You are a designer operating strictly through the {lens_name} lens: {lens_move}. Given the brief below, "
                    f"produce ONE fully realized concept that could only have come from this lens — if it would work equally well "
                    f"relabeled under a different lens, you have failed. Include: title, main visual idea, composition, lighting, "
                    f"background, color palette, typography direction, creative twist, designer execution notes, reference image prompt, "
                    f"a reasoning chain (tension addressed -> design decision -> consequence -> risk flag), a primary metaphor plus two "
                    f"rejected alternatives with one-line reasons, camera-language/lighting-reasoning/material-reasoning sub-panels, "
                    f"and one anti-pattern line. Return JSON matching the Concept schema exactly."
                )
                
                user_msg = (
                    f"Brief Details:\n{json.dumps(brief)}\n\n"
                    f"Clarifying Q&A:\n{json.dumps(answers)}\n\n"
                    f"Justification for applying this lens:\n{lens_item.justification}"
                )
                
                start_time = time.time()
                # Run two turns in a chat session for reflection
                try:
                    model = genai.GenerativeModel(
                        model_name=self.pro_model_name,
                        system_instruction=sys_prompt
                    )
                    chat = model.start_chat()
                    
                    # Map creative distance (1-5) to temperature (0.3 - 0.9)
                    generation_temp = round(0.3 + (creative_distance - 1) * 0.15, 2)
                    config = GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=ConceptGenerationResponse,
                        temperature=generation_temp
                    )
                    # Turn 1: Draft
                    response1 = await asyncio.to_thread(chat.send_message, user_msg, generation_config=config)
                    
                    # Turn 2: Reflection
                    reflection_prompt = (
                        f"Reread your concept. Does it actually embody the {lens_name} lens, or did it drift into generic "
                        f"AI-mood-board language? If any field is generic, rewrite ONLY that field. Return the full, "
                        f"corrected JSON object."
                    )
                    response2 = await asyncio.to_thread(chat.send_message, reflection_prompt, generation_config=config)
                    
                    concept_data = json.loads(response2.text)
                    concept_data["concept_number"] = idx
                    concept_data["lens_id"] = lens_item.lens_id
                    concept_data["style_category"] = lens_name
                    concept_data["lens_applied"] = lens_name
                    
                    # Cache in Redis with short TTL (10 minutes)
                    if redis_client:
                        try:
                            redis_client.setex(cache_key, 600, json.dumps(concept_data))
                        except Exception as cx:
                            logger.warning(f"Concept cache write failed: {cx}")

                    latency = int((time.time() - start_time) * 1000)
                    self._log_provider_call(f"concept_gen_lens_{lens_name}", "success", latency)
                    return {"chat": chat, "data": concept_data, "lens_id": lens_item.lens_id}
                except Exception as e:
                    logger.error(f"Error generating concept for lens {lens_name}: {str(e)}")
                    # Fallback metadata
                    latency = int((time.time() - start_time) * 1000)
                    self._log_provider_call(f"concept_gen_lens_{lens_name}", "failure", latency, error_msg=str(e))
                    fallback_data = {
                        "concept_number": idx,
                        "title": f"The {lens_name} Proposal",
                        "main_visual_idea": f"Visual approach using lens {lens_name}.",
                        "composition": "Balanced structural placement.",
                        "lighting": "Atmospheric casting.",
                        "background": "Textured background matching visual weight.",
                        "color_palette": ["#111111", "#cccccc"],
                        "typography_direction": "Clean geometric sans.",
                        "creative_twist": "Unexpected alignment.",
                        "designer_execution_notes": "Pay attention to raw constraints.",
                        "reference_image_prompt": f"Reference only: graphic representation of {lens_name} style.",
                        "lens_id": lens_item.lens_id,
                        "style_category": lens_name,
                        "lens_applied": lens_name
                    }
                    return {"chat": None, "data": fallback_data, "lens_id": lens_item.lens_id}

        tasks = [generate_single_concept(lens_item, i) for i, lens_item in enumerate(cd_selection.selected_lenses, 1)]
        generated_results = await asyncio.gather(*tasks)

        # 4. Critic Panel (1 batched call, flash)
        critic_instruction = (
            "You are part of SIFS Creative Reasoning Engine, a multi-agent creative-direction system for "
            "professional graphic designers. Your outputs are never final artwork. You produce reasoning and critique.\n\n"
            "You are a two-role Critic Panel: a production-realist Feasibility/Producer, and an "
            "Originality Critic checking against the trend-saturation data provided. You'll receive 10 "
            "draft concepts together. For each: feasibility score 1-5 with one-line reason, and flag if it "
            "resembles an oversaturated trend or another concept in this set. Write ONE paragraph of "
            "cross-concept critique. Where you substantively disagree with a concept's own premise, say so "
            "explicitly and mark it as dissent — do not soften disagreement into consensus. Return JSON matching CriticPanelResponse."
        )
        concepts_json_str = json.dumps([r["data"] for r in generated_results], indent=2)
        trend_context = "Current oversaturated trends: liquid metal typography, neon sunset gradients, glassmorphism UI card mockups, generic clay 3D illustrations."
        
        critic_response = self._call_gemini_structured(
            model_name=self.flash_model_name,
            prompt=f"Concepts set:\n{concepts_json_str}\n\nTrend Context:\n{trend_context}",
            schema=CriticPanelResponse,
            system_instruction=critic_instruction,
            task_type="critic_panel"
        )

        # Attach critic scores and dissent notes
        for item in critic_response.per_concept:
            idx = item.concept_number
            for r in generated_results:
                if r["data"]["concept_number"] == idx:
                    # Normalize feasibility to float
                    r["data"]["feasibility_score"] = float(item.feasibility_score)
                    r["data"]["feasibility_reason"] = item.feasibility_reason
                    r["data"]["critic_notes"] = [
                        {"agent_role": "Critic Panel", "note_text": n.note, "is_dissent": n.is_dissent}
                        for n in critic_response.dissent_notes if n.concept_number == idx
                    ]

        # 5. Creative Director Pass 3 (Rewrite instructions - up to 3)
        cd_rewrite_instruction = (
            "You are the Creative Director. Review these concepts and their Critic reviews. Select up to "
            "3 concepts that received heavy critique or have low feasibility/originality, and generate "
            "a specific rewrite instruction for them. Return JSON with the list of rewrites."
        )
        
        critique_summary = json.dumps(critic_response.model_dump(), indent=2)
        rewrites_response = self._call_gemini_structured(
            model_name=self.pro_model_name,
            prompt=f"Concepts:\n{concepts_json_str}\n\nCritiques:\n{critique_summary}",
            schema=CreativeDirectorRewriteResponse,
            system_instruction=cd_rewrite_instruction,
            task_type="cd_rewrites"
        )

        # 6. Apply rewrites (parallelized via asyncio)
        async def rewrite_single_concept(rewrite_item: RewriteInstructionItem):
            session_dict = next((r for r in generated_results if r["data"]["concept_number"] == rewrite_item.concept_number), None)
            if session_dict and session_dict["chat"]:
                chat = session_dict["chat"]
                rewrite_prompt = (
                    f"Rewrite this concept based on the Creative Director instruction:\n"
                    f"'{rewrite_item.instruction}'\n"
                    f"Return the complete corrected JSON matching the Concept schema."
                )
                start_time = time.time()
                try:
                    config = GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=ConceptGenerationResponse,
                        temperature=0.6
                    )
                    response = await asyncio.to_thread(chat.send_message, rewrite_prompt, generation_config=config)
                    updated_data = json.loads(response.text)
                    
                    # Update cache key
                    cache_key = f"concept_cache:{brief_hash}:{session_dict['lens_id']}"
                    if redis_client:
                        try:
                            redis_client.setex(cache_key, 600, json.dumps(updated_data))
                        except Exception as cx:
                            logger.warning(f"Concept cache write failed: {cx}")

                    # Restore structure fields
                    updated_data["concept_number"] = rewrite_item.concept_number
                    updated_data["lens_id"] = session_dict["lens_id"]
                    updated_data["style_category"] = session_dict["data"].get("style_category")
                    updated_data["lens_applied"] = session_dict["data"].get("lens_applied")
                    updated_data["feasibility_score"] = session_dict["data"].get("feasibility_score", 3.0)
                    updated_data["feasibility_reason"] = session_dict["data"].get("feasibility_reason", "")
                    updated_data["critic_notes"] = session_dict["data"].get("critic_notes", [])
                    
                    session_dict["data"] = updated_data
                    latency = int((time.time() - start_time) * 1000)
                    self._log_provider_call("concept_rewrite", "success", latency)
                except Exception as e:
                    logger.error(f"Error parsing rewritten concept: {str(e)}")
                    latency = int((time.time() - start_time) * 1000)
                    self._log_provider_call("concept_rewrite", "failure", latency, error_msg=str(e))

        if rewrites_response.rewrites:
            rewrite_tasks = [rewrite_single_concept(rw) for rw in rewrites_response.rewrites[:3]]
            await asyncio.gather(*rewrite_tasks)

        # Flatten concepts out
        final_concepts = [r["data"] for r in generated_results]

        # 7. Missing Opportunity Finder (1 call, flash)
        mo_instruction = (
            "You are the Missing Opportunity Finder. Examine the original brief and the 10 generated concepts. "
            "Identify what design directions or creative strategies were NOT covered but represent solid possibilities "
            "for the project. Return a JSON structure containing a comprehensive analysis report."
        )
        mo_response = self._call_gemini_structured(
            model_name=self.flash_model_name,
            prompt=f"Brief:\n{brief_data}\n\nGenerated Concepts:\n{json.dumps(final_concepts, indent=2)}",
            schema=MissingOpportunityResponse,
            system_instruction=mo_instruction,
            task_type="missing_opportunities"
        )

        return {
            "project_title": brief.get("main_subject", "Expanded Project"),
            "brief_summary": f"Creative Reasoning concepts generated for: '{brief.get('main_subject')}' brief.",
            "concepts": final_concepts,
            "missing_opportunity_report": mo_response.report_text
        }

    def regenerate_concept(self, brief: Dict[str, Any], original_concept: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        """Regenerate a single concept based on designer instructions."""
        start_time = time.time()
        try:
            model = genai.GenerativeModel(
                model_name=self.pro_model_name,
                system_instruction="You are SIFS design assistant. Modify the concept as requested, returning the full updated JSON object."
            )
            prompt = (
                f"Original Brief:\n{json.dumps(brief)}\n\n"
                f"Original Concept:\n{json.dumps(original_concept)}\n\n"
                f"Modification Instruction:\n{instruction}\n\n"
                f"Return the full corrected concept matching the Concept schema structure."
            )
            config = GenerationConfig(
                response_mime_type="application/json",
                response_schema=ConceptGenerationResponse,
                temperature=0.7
            )
            response = model.generate_content(prompt, generation_config=config)
            validated = json.loads(response.text)
            
            latency = int((time.time() - start_time) * 1000)
            self._log_provider_call("regenerate_concept", "success", latency)
            return validated
        except Exception as e:
            latency = int((time.time() - start_time) * 1000)
            self._log_provider_call("regenerate_concept", "failure", latency, error_msg=str(e))
            logger.error(f"Regenerate concept call failed: {e}")
            return original_concept

    def combine_concepts(self, brief: Dict[str, Any], concept_a: Dict[str, Any], concept_b: Dict[str, Any], custom_instruction: Optional[str] = None) -> Dict[str, Any]:
        """Combine two concepts into one new hybrid concept."""
        start_time = time.time()
        try:
            model = genai.GenerativeModel(
                model_name=self.pro_model_name,
                system_instruction="You are SIFS senior design director. Fuse the visual and conceptual elements of Concept A and Concept B."
            )
            prompt = (
                f"Brief:\n{json.dumps(brief)}\n\n"
                f"Concept A:\n{json.dumps(concept_a)}\n\n"
                f"Concept B:\n{json.dumps(concept_b)}\n\n"
                f"Custom Fusion Instruction:\n{custom_instruction or 'None'}\n\n"
                f"Generate a hybrid concept matching the Concept schema."
            )
            config = GenerationConfig(
                response_mime_type="application/json",
                response_schema=ConceptGenerationResponse,
                temperature=0.8
            )
            response = model.generate_content(prompt, generation_config=config)
            validated = json.loads(response.text)
            
            latency = int((time.time() - start_time) * 1000)
            self._log_provider_call("combine_concepts", "success", latency)
            return validated
        except Exception as e:
            latency = int((time.time() - start_time) * 1000)
            self._log_provider_call("combine_concepts", "failure", latency, error_msg=str(e))
            logger.error(f"Combine concepts call failed: {e}")
            return {
                "title": f"Hybrid: {concept_a.get('title')} + {concept_b.get('title')}",
                "main_visual_idea": f"Combined visual elements of {concept_a.get('title')} and {concept_b.get('title')}.",
                "composition": f"Fused: {concept_a.get('composition')} + {concept_b.get('composition')}",
                "lighting": concept_a.get("lighting"),
                "background": concept_b.get("background"),
                "color_palette": list(set(concept_a.get("color_palette", []) + concept_b.get("color_palette", []))),
                "typography_direction": concept_a.get("typography_direction"),
                "creative_twist": "Tactile blend of features.",
                "designer_execution_notes": "Pay attention to merging layout weights.",
                "reference_image_prompt": f"Reference only: Hybrid image merging {concept_a.get('title')} and {concept_b.get('title')}"
            }

    def test_connection(self) -> bool:
        start_time = time.time()
        try:
            model = genai.GenerativeModel(self.flash_model_name)
            response = model.generate_content("hello")
            success = len(response.text) > 0
            latency = int((time.time() - start_time) * 1000)
            self._log_provider_call("test_connection", "success" if success else "failure", latency)
            return success
        except Exception as e:
            latency = int((time.time() - start_time) * 1000)
            self._log_provider_call("test_connection", "failure", latency, error_msg=str(e))
            logger.error(f"Gemini connection test failed: {str(e)}")
            return False
