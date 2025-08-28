from pydantic import BaseModel, Field
from typing import Optional, List

class Accident(BaseModel):
    """Accident details"""
    date: str = Field(description="Accident date")
    time: Optional[str] = Field(default=None, description="Accident time")
    city: str = Field(description="City where accident occurred")
    location: str = Field(description="Exact accident location")
    number_of_vehicles_involved: str = Field(description="Number of vehicles involved")
    police_report_made: Optional[str] = Field(default=None, description="Whether a police report was made (Yes/No)")
    police_agency: Optional[str] = Field(default=None, description="Police agency if report was made")
    description: Optional[str] = Field(default=None, description="Accident description")