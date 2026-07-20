import json
from typing import List, Optional, Dict, Any
from sqlmodel import Session, select
from app.models.project import Project
from app.models.creative_brief import CreativeBrief
from app.models.clarifying_question import ClarifyingQuestion
from app.models.concept import Concept
from app.models.style_profile import StyleProfile
from app.models.reasoning import ReasoningChain, ConceptScore, CriticNote
from app.services.provider_router import get_llm_provider
from app.services.diversity_service import DiversityService
from app.schemas.concept import ConceptGenerateRequest, ConceptRegenerateRequest, ConceptCombineRequest

class ConceptService:
    @classmethod
    def generate_concepts(cls, session: Session, request: ConceptGenerateRequest) -> List[Dict[str, Any]]:
        project = session.get(Project, request.project_id)
        if not project:
            raise ValueError("Project not found")

        brief = session.exec(select(CreativeBrief).where(CreativeBrief.project_id == project.id)).first()
        if not brief:
            raise ValueError("Creative brief must be extracted first")

        # Fetch questions/answers
        questions = session.exec(select(ClarifyingQuestion).where(
            ClarifyingQuestion.project_id == project.id, 
            ClarifyingQuestion.answer != None
        )).all()
        answers = [{"question": q.question, "answer": q.answer} for q in questions]

        # Fetch style profile
        style_profile_data = None
        profile = session.exec(select(StyleProfile).where(StyleProfile.user_id == (project.user_id or 1))).first()
        if profile:
            style_profile_data = profile.model_dump()

        brief_dict = {
            "main_subject": brief.main_subject,
            "design_type": brief.design_type,
            "target_audience": brief.target_audience,
            "mood": brief.mood,
            "colors": brief.colors,
            "fixed_elements": brief.fixed_elements,
            "flexible_elements": brief.flexible_elements,
            "avoid_elements": brief.avoid_elements,
            "tensions": brief.tensions
        }

        provider = get_llm_provider()
        concepts_data = provider.generate_concepts(brief_dict, answers, style_profile=style_profile_data)

        # Clear existing concepts and related reasoning tables
        existing_concepts = session.exec(select(Concept).where(Concept.project_id == project.id)).all()
        for ec in existing_concepts:
            # Delete chains, scores, notes
            session.exec(select(ReasoningChain).where(ReasoningChain.concept_id == ec.id)).delete()
            session.exec(select(ConceptScore).where(ConceptScore.concept_id == ec.id)).delete()
            session.exec(select(CriticNote).where(CriticNote.concept_id == ec.id)).delete()
            session.delete(ec)

        # Insert new concepts & V2 details
        db_concepts = []
        for c_data in concepts_data["concepts"]:
            db_c = Concept(
                project_id=project.id,
                concept_number=c_data["concept_number"],
                title=c_data["title"],
                style_category=c_data["style_category"],
                main_visual_idea=c_data["main_visual_idea"],
                composition=c_data["composition"],
                lighting=c_data["lighting"],
                background=c_data["background"],
                color_palette=c_data["color_palette"],
                typography_direction=c_data["typography_direction"],
                creative_twist=c_data["creative_twist"],
                designer_execution_notes=c_data["designer_execution_notes"],
                reference_image_prompt=c_data["reference_image_prompt"],
                reference_only_notice=c_data["reference_only_notice"],
                # V2 additions
                metaphor=c_data.get("metaphor", ""),
                rejected_metaphor_1=c_data.get("rejected_metaphor_1", ""),
                rejected_metaphor_2=c_data.get("rejected_metaphor_2", ""),
                camera_language=c_data.get("camera_language", ""),
                lighting_reasoning=c_data.get("lighting_reasoning", ""),
                material_reasoning=c_data.get("material_reasoning", ""),
                anti_pattern=c_data.get("anti_pattern", ""),
                status="active"
            )
            session.add(db_c)
            session.commit()
            session.refresh(db_c)
            db_concepts.append(db_c)

            # Insert Reasoning Chain
            db_chain = ReasoningChain(
                concept_id=db_c.id,
                lens_applied=c_data.get("lens_applied", db_c.style_category),
                tension_identified=c_data.get("tension_identified", "Adhering to brief vs exploration constraints"),
                design_decision=c_data.get("design_decision", "Synthesized custom aesthetic values"),
                consequence=c_data.get("consequence", "Provided unique visual direction"),
                risk_flag=c_data.get("risk_flag", "Creative boundary push")
            )
            session.add(db_chain)

            # Insert Scores
            db_score = ConceptScore(
                concept_id=db_c.id,
                novelty_score=c_data.get("novelty_score", 0.8),
                feasibility_score=c_data.get("feasibility_score", 0.75),
                feasibility_reason=c_data.get("feasibility_reason", "Executable with normal design resources"),
                lens_fidelity_score=c_data.get("lens_fidelity_score", 0.9)
            )
            session.add(db_score)

            # Insert Critic Notes (if any)
            for note in c_data.get("critic_notes", []):
                db_note = CriticNote(
                    concept_id=db_c.id,
                    agent_role=note.get("agent_role", "Critic"),
                    note_text=note.get("note_text", ""),
                    is_dissent=note.get("is_dissent", False)
                )
                session.add(db_note)
            
            session.commit()

        # Update diversity score using existing service
        try:
            scores = DiversityService.calculate_pairwise_similarities(db_concepts)
            for c in db_concepts:
                max_similarity = 0.0
                for w in scores.pairwise_warnings:
                    if w.concept_a == c.concept_number or w.concept_b == c.concept_number:
                        max_similarity = max(max_similarity, w.similarity)
                c.diversity_score = round(1.0 - max_similarity, 2)
                session.add(c)
            session.commit()
        except Exception:
            pass

        project.status = "concepts_generated"
        session.add(project)
        session.commit()

        return cls.get_concepts_by_project(session, project.id)

    @classmethod
    def get_concepts_by_project(cls, session: Session, project_id: int) -> List[Dict[str, Any]]:
        concepts = session.exec(
            select(Concept).where(Concept.project_id == project_id).order_by(Concept.concept_number.asc())
        ).all()
        
        enriched = []
        for c in concepts:
            chain = session.exec(select(ReasoningChain).where(ReasoningChain.concept_id == c.id)).first()
            score = session.exec(select(ConceptScore).where(ConceptScore.concept_id == c.id)).first()
            notes = session.exec(select(CriticNote).where(CriticNote.concept_id == c.id)).all()

            c_dict = c.model_dump()
            c_dict["reasoning_chain"] = chain.model_dump() if chain else None
            c_dict["scores"] = score.model_dump() if score else None
            c_dict["critic_notes"] = [n.model_dump() for n in notes]
            enriched.append(c_dict)

        return enriched

    @classmethod
    def regenerate_concept(cls, session: Session, request: ConceptRegenerateRequest) -> Dict[str, Any]:
        concept = session.get(Concept, request.concept_id)
        if not concept:
            raise ValueError("Concept not found")

        project = session.get(Project, concept.project_id)
        brief = session.exec(select(CreativeBrief).where(CreativeBrief.project_id == project.id)).first()
        brief_dict = {
            "main_subject": brief.main_subject,
            "design_type": brief.design_type
        }

        # Pack original concept fields
        concept_dict = concept.model_dump()
        
        provider = get_llm_provider()
        new_data = provider.regenerate_concept(brief_dict, concept_dict, request.instruction)

        # Update Concept details
        for key in ["title", "main_visual_idea", "composition", "lighting", "background", "color_palette", 
                    "typography_direction", "creative_twist", "designer_execution_notes", "reference_image_prompt",
                    "metaphor", "rejected_metaphor_1", "rejected_metaphor_2", "camera_language", 
                    "lighting_reasoning", "material_reasoning", "anti_pattern"]:
            if key in new_data:
                setattr(concept, key, new_data[key])
        session.add(concept)
        session.commit()

        # Update Reasoning Chain details
        chain = session.exec(select(ReasoningChain).where(ReasoningChain.concept_id == concept.id)).first()
        if chain:
            chain.design_decision = new_data.get("design_decision", chain.design_decision)
            chain.consequence = new_data.get("consequence", chain.consequence)
            session.add(chain)
            session.commit()

        # Re-fetch enriched
        return cls.get_enriched_concept(session, concept.id)

    @classmethod
    def combine_concepts(cls, session: Session, request: ConceptCombineRequest) -> Dict[str, Any]:
        concept_a = session.get(Concept, request.concept_a_id)
        concept_b = session.get(Concept, request.concept_b_id)
        if not concept_a or not concept_b:
            raise ValueError("One or both concepts not found")

        project = session.get(Project, concept_a.project_id)
        brief = session.exec(select(CreativeBrief).where(CreativeBrief.project_id == project.id)).first()

        brief_dict = {"main_subject": brief.main_subject}
        a_dict = concept_a.model_dump()
        b_dict = concept_b.model_dump()

        provider = get_llm_provider()
        combined_data = provider.combine_concepts(brief_dict, a_dict, b_dict, request.custom_instruction)

        # Determine next index
        existing = session.exec(select(Concept).where(Concept.project_id == project.id)).all()
        next_num = max(c.concept_number for c in existing) + 1 if existing else 11

        db_c = Concept(
            project_id=project.id,
            concept_number=next_num,
            title=combined_data.get("title", f"Hybrid concept {next_num}"),
            style_category="Hybrid Concept",
            main_visual_idea=combined_data.get("main_visual_idea", ""),
            composition=combined_data.get("composition", ""),
            lighting=combined_data.get("lighting", ""),
            background=combined_data.get("background", ""),
            color_palette=combined_data.get("color_palette", []),
            typography_direction=combined_data.get("typography_direction", ""),
            creative_twist=combined_data.get("creative_twist", ""),
            designer_execution_notes=combined_data.get("designer_execution_notes", ""),
            reference_image_prompt=combined_data.get("reference_image_prompt", ""),
            reference_only_notice="Reference only - final artwork belongs to the designer.",
            metaphor=combined_data.get("metaphor", "Fitted blend"),
            rejected_metaphor_1=combined_data.get("rejected_metaphor_1", ""),
            rejected_metaphor_2=combined_data.get("rejected_metaphor_2", ""),
            camera_language=combined_data.get("camera_language", ""),
            lighting_reasoning=combined_data.get("lighting_reasoning", ""),
            material_reasoning=combined_data.get("material_reasoning", ""),
            anti_pattern=combined_data.get("anti_pattern", ""),
            status="active"
        )
        session.add(db_c)
        session.commit()
        session.refresh(db_c)

        # Save Reasoning Chain
        db_chain = ReasoningChain(
            concept_id=db_c.id,
            lens_applied="Hybrid",
            tension_identified=combined_data.get("tension_identified", "Visual cohesion blend"),
            design_decision=combined_data.get("design_decision", "Merged visual properties"),
            consequence="Formed hybrid aesthetic proposal",
            risk_flag=combined_data.get("risk_flag", "")
        )
        session.add(db_chain)

        # Save Scores
        db_score = ConceptScore(
            concept_id=db_c.id,
            novelty_score=0.85,
            feasibility_score=0.75,
            feasibility_reason="Fusing two styles requires active layout decisions",
            lens_fidelity_score=0.9
        )
        session.add(db_score)
        session.commit()

        return cls.get_enriched_concept(session, db_c.id)

    @classmethod
    def get_enriched_concept(cls, session: Session, concept_id: int) -> Dict[str, Any]:
        c = session.get(Concept, concept_id)
        if not c:
            raise ValueError("Concept not found")
        
        chain = session.exec(select(ReasoningChain).where(ReasoningChain.concept_id == c.id)).first()
        score = session.exec(select(ConceptScore).where(ConceptScore.concept_id == c.id)).first()
        notes = session.exec(select(CriticNote).where(CriticNote.concept_id == c.id)).all()

        c_dict = c.model_dump()
        c_dict["reasoning_chain"] = chain.model_dump() if chain else None
        c_dict["scores"] = score.model_dump() if score else None
        c_dict["critic_notes"] = [n.model_dump() for n in notes]
        return c_dict

    @staticmethod
    def update_concept_status(session: Session, concept_id: int, status: str) -> Optional[Dict[str, Any]]:
        db_c = session.get(Concept, concept_id)
        if not db_c:
            return None
        db_c.status = status
        session.add(db_c)
        session.commit()
        session.refresh(db_c)
        
        # Load relations
        chain = session.exec(select(ReasoningChain).where(ReasoningChain.concept_id == db_c.id)).first()
        score = session.exec(select(ConceptScore).where(ConceptScore.concept_id == db_c.id)).first()
        notes = session.exec(select(CriticNote).where(CriticNote.concept_id == db_c.id)).all()

        c_dict = db_c.model_dump()
        c_dict["reasoning_chain"] = chain.model_dump() if chain else None
        c_dict["scores"] = score.model_dump() if score else None
        c_dict["critic_notes"] = [n.model_dump() for n in notes]
        return c_dict
