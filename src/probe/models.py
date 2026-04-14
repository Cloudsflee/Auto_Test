from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProbeAssertion:
    assert_type: str
    expect: Any = None
    flags: str = ""
    negate: bool = False
    field_path: str = ""
    ext: list[str] = field(default_factory=list)
    gte: float | int | None = None


@dataclass
class ProbeTarget:
    source: str
    turn: int = 0


@dataclass
class ProbeSpec:
    probe_id: str
    probe_type: str
    judge_mode: str
    target: ProbeTarget
    assertions: list[ProbeAssertion]
    weight: float = 1.0
    critical: bool = False
    priority: str = "normal"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    evidence_policy: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeDataset:
    dataset_id: str
    dataset_version: str
    description: str
    owner: str
    probes: list[ProbeSpec]
