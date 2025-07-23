from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Paper(Base):
    """SQLAlchemy model for academic papers"""
    __tablename__ = 'papers'

    id = Column(Integer, primary_key=True)
    title = Column(String, unique=True, nullable=False)
    authors = Column(String)
    abstract = Column(String)
    year = Column(Integer)
    citations = Column(Integer)
    url = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Define indexes
    __table_args__ = (
        Index('idx_papers_title', 'title'),
        Index('idx_papers_year', 'year'),
        Index('idx_papers_citations', 'citations'),
    )

    def __repr__(self):
        return f"<Paper(title='{self.title}', year={self.year}, citations={self.citations})>" 