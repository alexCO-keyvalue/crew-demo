from pydantic import BaseModel, Field
from typing import List, Optional

class TripDetails(BaseModel):
    """Model for extracted trip details"""
    destination: Optional[str] = Field(None, description="The destination city or location for the trip")
    duration: Optional[str] = Field(None, description="Duration of the trip in days")
    start_date: Optional[str] = Field(None, description="Start date of the trip")
    budget: Optional[str] = Field(None, description="Budget range")
    interests: List[str] = Field(default_factory=list, description="List of interests or preferences mentioned")
    group_size: Optional[int] = Field(None, description="Number of people traveling")
    accommodation_type: Optional[str] = Field(None, description="Preferred accommodation type if mentioned")

class Attraction(BaseModel):
    """Model for a single attraction"""
    name: str = Field(..., description="Name of the attraction")
    description: str = Field(..., description="Description of the attraction")
    location: str = Field(..., description="Location/address of the attraction")
    opening_hours: Optional[str] = Field(None, description="Opening hours if available")
    estimated_visit_time: Optional[str] = Field(None, description="Estimated time needed to visit")
    category: Optional[str] = Field(None, description="Category of attraction (museum, park, restaurant, etc.)")
    rating: Optional[float] = Field(None, description="Rating if available")

class AttractionsSearchResult(BaseModel):
    """Model for attractions search results"""
    destination: str = Field(..., description="The destination that was searched")
    attractions: List[Attraction] = Field(..., description="List of found attractions")
    total_found: int = Field(..., description="Total number of attractions found")
    search_date: str = Field(..., description="Date when the search was performed") 