"""Data models and schemas for validation."""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
import re

class NewsArticle(BaseModel):
    """Schema for validating news articles from MongoDB."""
    source: str = Field(..., description="Source of the news (e.g., VnExpress)")
    title: str = Field(..., description="Title of the news article")
    content: str = Field(..., description="The main content or first few paragraphs")
    date: str = Field(..., description="ISO-8601 date string (e.g., 2024-03-08T02:00:00.000Z)")
    error: Optional[str] = Field(default=None, description="Error message if extraction failed")

    @field_validator('date')
    @classmethod
    def validate_iso_date(cls, v: str) -> str:
        """Ensure date follows a standard ISO-8601 format."""
        # Simple regex for YYYY-MM-DDTHH:MM:SS...
        iso_regex = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        if not re.match(iso_regex, v):
            raise ValueError(f"Date '{v}' must follow ISO-8601 format (YYYY-MM-DDTHH:MM:SS)")
        return v
