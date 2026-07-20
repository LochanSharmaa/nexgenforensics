import json
from typing import Optional
from sqlmodel import Session, select
from app.models.project import Project
from app.models.creative_brief import CreativeBrief
from app.models.concept import Concept

class ExportService:
    @staticmethod
    def export_json(session: Session, project_id: int) -> str:
        project = session.get(Project, project_id)
        if not project:
            raise ValueError("Project not found")

        brief = session.exec(select(CreativeBrief).where(CreativeBrief.project_id == project_id)).first()
        concepts = session.exec(select(Concept).where(Concept.project_id == project_id).order_by(Concept.concept_number.asc())).all()

        data = {
            "project": {
                "id": project.id,
                "title": project.title,
                "raw_imagination": project.raw_imagination,
                "design_type": project.design_type,
                "status": project.status,
                "created_at": str(project.created_at)
            },
            "creative_brief": {
                "main_subject": brief.main_subject if brief else "",
                "design_type": brief.design_type if brief else "",
                "target_audience": brief.target_audience if brief else "",
                "mood": brief.mood if brief else [],
                "colors": brief.colors if brief else [],
                "fixed_elements": brief.fixed_elements if brief else [],
                "flexible_elements": brief.flexible_elements if brief else [],
                "avoid_elements": brief.avoid_elements if brief else []
            } if brief else None,
            "concepts": [
                {
                    "concept_number": c.concept_number,
                    "title": c.title,
                    "style_category": c.style_category,
                    "main_visual_idea": c.main_visual_idea,
                    "composition": c.composition,
                    "lighting": c.lighting,
                    "background": c.background,
                    "color_palette": c.color_palette,
                    "typography_direction": c.typography_direction,
                    "creative_twist": c.creative_twist,
                    "designer_execution_notes": c.designer_execution_notes,
                    "reference_image_prompt": c.reference_image_prompt,
                    "reference_only_notice": c.reference_only_notice
                } for c in concepts
            ]
        }
        return json.dumps(data, indent=2)

    @staticmethod
    def export_markdown(session: Session, project_id: int) -> str:
        project = session.get(Project, project_id)
        if not project:
            raise ValueError("Project not found")

        brief = session.exec(select(CreativeBrief).where(CreativeBrief.project_id == project_id)).first()
        concepts = session.exec(select(Concept).where(Concept.project_id == project_id).order_by(Concept.concept_number.asc())).all()

        md = f"# Concept Board: {project.title}\n\n"
        md += f"> **AI Creative Direction Assistant** - One imagination. Ten creative directions.\n"
        md += f"> Reference only - final artwork belongs to the designer.\n\n"
        
        md += f"## 1. User Imagination\n"
        md += f"```text\n{project.raw_imagination}\n```\n\n"

        if brief:
            md += f"## 2. Structured Creative Brief\n"
            md += f"- **Main Subject**: {brief.main_subject}\n"
            md += f"- **Design Type**: {brief.design_type}\n"
            md += f"- **Target Audience**: {brief.target_audience}\n"
            md += f"- **Mood**: {', '.join(brief.mood)}\n"
            md += f"- **Color Palette**: {', '.join(brief.colors)}\n"
            md += f"- **Must-Keep Elements**: {', '.join(brief.fixed_elements)}\n"
            md += f"- **Flexible Elements**: {', '.join(brief.flexible_elements)}\n"
            md += f"- **Avoid Elements**: {', '.join(brief.avoid_elements)}\n\n"

        md += f"## 3. Visual Concept Directions (Exactly 10 Style Categories)\n\n"
        for c in concepts:
            md += f"### Concept {c.concept_number}: {c.title}\n"
            md += f"- **Style Category**: {c.style_category}\n"
            md += f"- **Visual Metaphor & Core Idea**: {c.main_visual_idea}\n"
            md += f"- **Composition**: {c.composition}\n"
            md += f"- **Lighting Plan**: {c.lighting}\n"
            md += f"- **Background Details**: {c.background}\n"
            md += f"- **Color Scheme**: {', '.join(c.color_palette)}\n"
            md += f"- **Typography Direction**: {c.typography_direction}\n"
            md += f"- **Creative Twist**: {c.creative_twist}\n"
            md += f"- **Execution Notes for Designer**: *{c.designer_execution_notes}*\n"
            md += f"- **Image Generator Prompt**: `{c.reference_image_prompt}`\n"
            md += f"- *Notice: {c.reference_only_notice}*\n\n"
            md += f"---\n\n"

        return md
