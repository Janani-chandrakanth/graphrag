from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Step:
    """Represents a single workflow step."""
    id: str
    action: str
    actors: List[str] = field(default_factory=list)
    systems: List[str] = field(default_factory=list)
    data_objects: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class Feature:
    """A feature containing an ordered list of steps."""
    name: str
    steps: List[Step] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class WorkflowModel:
    """Top‑level model holding all extracted features."""
    features: List[Feature] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)
