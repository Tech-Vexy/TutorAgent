import os
import json
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, JSON, select, text
from sqlalchemy.dialects.postgresql import JSONB
from contextlib import asynccontextmanager
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/tutor_db")

# Fix for SQLAlchemy Async Engine with psycopg 3
# SQLAlchemy requires 'postgresql+psycopg://' scheme for psycopg 3 async
if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

# Connection string for psycopg (no asyncpg driver prefix usually needed for pool, but let's check)
# AsyncPostgresSaver uses psycopg 3.
DB_URI = os.getenv("DB_URI", "postgresql://postgres:postgres@localhost:5432/tutor_db")

# SQLAlchemy Setup
Base = declarative_base()
engine = create_async_engine(
    DATABASE_URL, 
    echo=False, 
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5
    }
)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

class StudentProfile(Base):
    __tablename__ = "student_profiles"

    student_id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=True)
    weak_subjects = Column(JSONB, default=list)  # List of strings
    learning_style = Column(String, default="Visual")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "student_id": self.student_id,
            "name": self.name,
            "weak_subjects": self.weak_subjects,
            "learning_style": self.learning_style
        }

async def init_db():
    """Initialize the database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_or_create_profile(student_id: str, name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get or create a student profile.
    This function is designed to be used as a tool by the agent.
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(StudentProfile).where(StudentProfile.student_id == student_id))
        profile = result.scalar_one_or_none()

        if not profile:
            profile = StudentProfile(student_id=student_id, name=name)
            session.add(profile)
            await session.commit()
            await session.refresh(profile)
        
        return profile.to_dict()

async def update_profile(student_id: str, **kwargs) -> Dict[str, Any]:
    """Update a student profile."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(StudentProfile).where(StudentProfile.student_id == student_id))
        profile = result.scalar_one_or_none()
        
        if profile:
            for key, value in kwargs.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
            await session.commit()
            await session.refresh(profile)
            return profile.to_dict()
        return None


async def upsert_thread_metadata(thread_id: str, user_id: str, title: Optional[str] = None) -> None:
    """Insert or update a user thread entry without relying on the psycopg pool."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                """
                INSERT INTO user_threads (thread_id, user_id, title)
                VALUES (:thread_id, :user_id, COALESCE(:title, ''))
                ON CONFLICT (thread_id) DO UPDATE
                SET
                    updated_at = CURRENT_TIMESTAMP,
                    title = CASE
                        WHEN COALESCE(:title, '') <> '' THEN :title
                        ELSE user_threads.title
                    END
                """
            ),
            {"thread_id": thread_id, "user_id": user_id, "title": title}
        )

@asynccontextmanager
async def get_postgres_checkpointer():
    """
    Yields an AsyncPostgresSaver connected to the database.
    Manages the connection pool lifecycle.
    """
    pool = AsyncConnectionPool(conninfo=DB_URI, open=False, kwargs={"autocommit": True})
    await pool.open()
    try:
        checkpointer = AsyncPostgresSaver(pool)
        # Ensure tables are created
        await checkpointer.setup()
        yield checkpointer
    finally:
        await pool.close()
