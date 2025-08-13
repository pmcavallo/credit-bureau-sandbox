from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class BureauRecord(BaseModel):
    customer_id: str
    snapshot_date: str
    income: float
    utilization_rate: float
    num_trades: int
    delinq_12m: int
    inquiries_6m: int
    dti: float
    cltv: float
    employment_status: str
    state: str
    age: int
    tenure_months: int

class DecisionRequest(BaseModel):
    records: List[BureauRecord]

class DecisionResponse(BaseModel):
    model_version: str
    scores: Dict[str, float]  # customer_id -> PD score
    reason_codes: Dict[str, List[str]]  # customer_id -> reason codes
