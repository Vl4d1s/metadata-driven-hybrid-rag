from pydantic import BaseModel, Field
from typing import Optional, List

class Owner(BaseModel):
    """Car owner information (no vehicle/license fields)"""
    first_name: str = Field(description="The owner's first name")
    last_name: str = Field(description="The owner's last name")
    id_number: str = Field(description="The owner's ID number")
    phone: str = Field(description="Mobile phone number")
    date_of_birth: Optional[str] = Field(default=None, description="Date of birth (YYYY-MM-DD)")
    zip_code: Optional[str] = Field(default=None, description="ZIP or postal code")
    city: str = Field(description="City of residence")
    street: Optional[str] = Field(default=None, description="Street name")
    house_number: Optional[str] = Field(default=None, description="House number")