from sqlalchemy import (
    create_engine, Column, String, Integer,
    Text, DateTime, JSON, ForeignKey, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
from config import config

Base = declarative_base()


class Call(Base):
    """One row per ingested transcript file."""
    __tablename__ = "calls"

    call_id         = Column(String, primary_key=True)   # e.g. "2_pricing_call"
    filename        = Column(String, nullable=False)
    ingested_at     = Column(DateTime, default=datetime.utcnow)
    duration_sec    = Column(Integer, nullable=True)     # parsed from last timestamp
    participants    = Column(JSON, default=list)         # ["Jordan (AE)", "Priya (RevOps)"]
    raw_text        = Column(Text, nullable=False)

    chunks          = relationship("Chunk", back_populates="call", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "call_id":      self.call_id,
            "filename":     self.filename,
            "ingested_at":  self.ingested_at.isoformat() if self.ingested_at else None,
            "duration_sec": self.duration_sec,
            "participants": self.participants,
            "chunk_count":  len(self.chunks),
        }


class Chunk(Base):
    """
    One row per transcript chunk (speaker-turn-aware).
    LLM assigns speaker_role at ingestion time.
    """
    __tablename__ = "chunks"

    chunk_id        = Column(String, primary_key=True)   # uuid
    call_id         = Column(String, ForeignKey("calls.call_id"), nullable=False)
    speaker         = Column(String, nullable=False)     # "Priya (RevOps Director)"
    speaker_role    = Column(String, nullable=False)     # "ae"|"prospect"|"se"|"ciso"|"other"
    start_time      = Column(String, nullable=False)     # "MM:SS"
    end_time        = Column(String, nullable=True)
    text            = Column(Text, nullable=False)
    embedding_id    = Column(Integer, nullable=True)     # FAISS index position

    call            = relationship("Call", back_populates="chunks")

    def to_dict(self) -> dict:
        return {
            "chunk_id":     self.chunk_id,
            "call_id":      self.call_id,
            "speaker":      self.speaker,
            "speaker_role": self.speaker_role,
            "start_time":   self.start_time,
            "end_time":     self.end_time,
            "text":         self.text,
            "citation":     f"[{self.call_id} | {self.start_time} | {self.speaker}]",
        }


# ── Engine + Session ──────────────────────────────────────────────────────────

engine = create_engine(
    f"sqlite:///{config.DB_PATH}",
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    """Create all tables. Safe to call multiple times."""
    Base.metadata.create_all(bind=engine)


def get_session():
    """Context-managed session for use in tools."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()