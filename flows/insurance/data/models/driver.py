from pydantic import BaseModel, Field
from typing import Optional, List

class Driver(BaseModel):
    """Driver information including personal, license, and vehicle details"""
    first_name: str = Field(description="The driver's first name")
    last_name: str = Field(description="The driver's last name")
    id_number: str = Field(description="The driver's ID number")
    phone: str = Field(description="Mobile phone number")
    date_of_birth: str = Field(description="Date of birth (YYYY-MM-DD)")
    zip_code: Optional[str] = Field(default=None, description="ZIP or postal code")
    city: str = Field(description="City of residence")
    street: Optional[str] = Field(default=None, description="Street name")
    house_number: Optional[str] = Field(default=None, description="House number")
    license_number: str = Field(description="Driver’s license number")
    license_plate: str = Field(description="License plate number")
    vehicle_make_and_model: str = Field(description="Vehicle make (brand) and model")
    vehicle_year: str = Field(description="Vehicle manufacturing year (as string)")
