from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


LLM_SUBJECTIVE_DIMENSIONS = [
    "accuracy",
    "context_awareness",
    "artifact_trail",
    "completeness",
    "continuity",
    "instruction_following",
]


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
class ProbeLLMJudgeSpec:
    rubric_id: str = "probe_subjective_v1"
    pass_threshold_0_5: float = 3.2
    dimensions: list[str] = field(default_factory=lambda: list(LLM_SUBJECTIVE_DIMENSIONS))
    turn_window_start: int = 0
    turn_window_end: int = 0
    require_evidence_paths: bool = False


@dataclass
class ProbeLLMJudgeConfig:
    enabled: bool = False
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    timeout_sec: int = 45
    repeats: int = 3
    max_retries: int = 2
    fail_open: bool = False
    system_prompt_path: str = ""
    user_prompt_path: str = ""


@dataclass
class ProbeSpec:
    probe_id: str
    probe_type: str
    judge_mode: str
    target: ProbeTarget
    assertions: list[ProbeAssertion]
    llm_judge: ProbeLLMJudgeSpec | None = None
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
