from pydantic import BaseModel, EmailStr
from datetime import datetime


class OrganizationCreate(BaseModel):
    name: str
    description: str | None = None
    industry: str | None = None
    contact_email: EmailStr | None = None


class OrganizationResponse(BaseModel):
    id: int
    name: str
    description: str | None = None
    industry: str | None = None
    contact_email: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
