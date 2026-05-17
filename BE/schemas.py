from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ModelVersionBase(BaseModel):
    version_name: str
    model_path: str
    is_active: bool = False

class ModelVersionCreate(ModelVersionBase):
    pass

class ModelVersionResponse(ModelVersionBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class TrainingRunBase(BaseModel):
    epochs: int
    image_size: int
    dataset_count: int
    status: str

class TrainingRunCreate(TrainingRunBase):
    model_version_id: int

class TrainingRunResponse(TrainingRunBase):
    id: int
    model_version_id: int
    precision: Optional[float] = None
    recall: Optional[float] = None
    map50: Optional[float] = None
    started_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True
