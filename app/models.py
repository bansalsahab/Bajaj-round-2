from pydantic import BaseModel
from typing import List, Optional

class LabTest(BaseModel):
    test_name: str
    test_value: str
    test_unit: str
    bio_reference_range: str
    lab_test_out_of_range: bool

class LabTestResponse(BaseModel):
    is_success: bool
    data: List[LabTest]
