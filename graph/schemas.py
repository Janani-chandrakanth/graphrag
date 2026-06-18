from pydantic import BaseModel


class Relationship(BaseModel):
    source: str
    relationship: str
    target: str