from typing import List, Optional
from sqlmodel import Session, select
from app.models.clarifying_question import ClarifyingQuestion
from app.models.creative_brief import CreativeBrief
from app.services.provider_router import get_llm_provider
from app.schemas.brief import ClarifyingQuestionAnswer

class QuestionService:
    @staticmethod
    def generate_questions(session: Session, project_id: int) -> List[ClarifyingQuestion]:
        # Fetch brief
        statement = select(CreativeBrief).where(CreativeBrief.project_id == project_id)
        brief = session.exec(statement).first()
        if not brief:
            raise ValueError("Creative brief must be extracted first")

        # Delete existing questions for this project
        stmt_del = select(ClarifyingQuestion).where(ClarifyingQuestion.project_id == project_id)
        existing = session.exec(stmt_del).all()
        for q in existing:
            session.delete(q)

        provider = get_llm_provider()
        brief_dict = {
            "main_subject": brief.main_subject,
            "design_type": brief.design_type,
            "target_audience": brief.target_audience,
            "mood": brief.mood,
            "colors": brief.colors,
            "fixed_elements": brief.fixed_elements,
            "flexible_elements": brief.flexible_elements,
            "avoid_elements": brief.avoid_elements
        }
        
        questions_text = provider.generate_questions(brief_dict)
        
        db_questions = []
        for q_text in questions_text:
            db_q = ClarifyingQuestion(
                project_id=project_id,
                question=q_text,
                skipped=False
            )
            session.add(db_q)
            db_questions.append(db_q)
            
        session.commit()
        for q in db_questions:
            session.refresh(q)
            
        return db_questions

    @staticmethod
    def get_questions_by_project(session: Session, project_id: int) -> List[ClarifyingQuestion]:
        statement = select(ClarifyingQuestion).where(ClarifyingQuestion.project_id == project_id)
        return session.exec(statement).all()

    @staticmethod
    def answer_question(
        session: Session, question_id: int, answer_in: ClarifyingQuestionAnswer
    ) -> Optional[ClarifyingQuestion]:
        db_q = session.get(ClarifyingQuestion, question_id)
        if not db_q:
            return None
        db_q.answer = answer_in.answer
        db_q.skipped = answer_in.skipped
        session.add(db_q)
        session.commit()
        session.refresh(db_q)
        return db_q
