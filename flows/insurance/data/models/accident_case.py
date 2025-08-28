
from pydantic import BaseModel
from typing import Optional
from .driver import Driver
from .owner import Owner
from .accident import Accident

class AccidentCase(BaseModel):
    """Full accident case including drivers, owner, and accident details"""
    driver_information: Driver
    owner_information: Owner
    other_driver_1_information: Driver
    other_driver_2_information: Optional[Driver] = None
    other_driver_3_information: Optional[Driver] = None
    accident_information: Accident