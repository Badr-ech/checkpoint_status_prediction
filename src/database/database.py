"""
Database connection and session management
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
import os
from dotenv import load_dotenv

from .models import Base

load_dotenv()

# Database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./checkpoint_data.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False  # Set to True for SQL debugging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully!")


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get database session
    
    Usage:
        @app.get("/items/")
        def read_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for database session
    
    Usage:
        with get_db_context() as db:
            checkpoints = db.query(Checkpoint).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def drop_all_tables():
    """Drop all tables (use with caution!)"""
    Base.metadata.drop_all(bind=engine)
    print("All tables dropped!")


def reset_database():
    """Drop and recreate all tables (use with caution!)"""
    drop_all_tables()
    init_db()
    print("Database reset complete!")
