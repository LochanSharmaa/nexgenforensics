from sqlmodel import create_engine, SQLModel, Session, select
from app.core.config import settings

# Create engine. If SQLite, configure it for multithreading
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args, echo=False)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
    seed_lenses()

def seed_lenses():
    from app.models.reasoning import Lens
    with Session(engine) as session:
        # Check if lenses already exist
        statement = select(Lens)
        existing = session.exec(statement).first()
        if existing:
            return
            
        # Seed the 12 dynamic creative reasoning lenses
        lenses = [
            Lens(name="Literalist", description="Executes the brief exactly as stated, at the highest possible craft level.", reasoning_move="The control group, pure execution of the user's brief.", risk_level=1),
            Lens(name="Contradiction", description="Finds the tension inside the brief itself and makes the tension the subject.", reasoning_move="Tension identification and dialectical visual synthesis.", risk_level=4),
            Lens(name="Subtraction", description="Removes the most obvious element the brief implies and designs around its absence.", reasoning_move="Obvious element deletion and visual vacuum centering.", risk_level=3),
            Lens(name="Wrong Medium", description="Answers the brief as if it were a different medium entirely.", reasoning_move="Format-transplanting logic (e.g. perfume poster as a record sleeve).", risk_level=4),
            Lens(name="Audience Inversion", description="Designs for the audience's opposite - who would this alienate, and is that person interesting?", reasoning_move="Antagonistic aesthetic targeting.", risk_level=3),
            Lens(name="Material Honesty", description="Starts from a physical material or process constraints and lets them drive the concept.", reasoning_move="Constraint-driven design based on physical craft limits (folding, ink bleeding).", risk_level=2),
            Lens(name="Narrative Compression", description="Treats the design as frame 1 of a story and reasons backward from an implied frame 2.", reasoning_move="Sequential frame compression and narrative tension.", risk_level=3),
            Lens(name="Emotional Extremity", description="Pushes one single emotion in the brief to its most extreme, uncomfortable version.", reasoning_move="Affective over-amplification.", risk_level=4),
            Lens(name="Cultural Counter-Signal", description="Deliberately signals against the category's expected visual codes.", reasoning_move="Category code negation (e.g., luxury perfume that refuses luxury layout codes).", risk_level=4),
            Lens(name="Constraint Injection", description="Self-imposes an arbitrary formal constraint and reasons through it.", reasoning_move="Artificial constraint injection (e.g. one-color, no images).", risk_level=3),
            Lens(name="Found Object", description="Reframes the subject as if discovered rather than designed - evidence, relic.", reasoning_move="Artifactual framing and mock-relic visual context.", risk_level=3),
            Lens(name="Restraint Maximalism", description="Says everything the brief wants with radically fewer decisions than seems possible.", reasoning_move="Extreme stylistic containment.", risk_level=2),
        ]
        for l in lenses:
            session.add(l)
        session.commit()

def get_session():
    with Session(engine) as session:
        yield session

