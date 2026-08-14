"""
Validated structured action plans for the Week 2 Brain/Body loop.

The Brain may be backed by a real LLM or by a deterministic mock, but the
Body only executes this schema. That keeps LLM output away from PyBullet and
gives the experiment harness a stable artifact to log and compare.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from alpha.core.entities import ActionType, BlockColor, Vec3


class ActionStep(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    action: ActionType
    target: Vec3 | None = None
    block: BlockColor | None = None

    @field_validator("target", mode="before")
    @classmethod
    def coerce_target(cls, value: Any) -> Vec3 | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError("target must be a 3-element list/tuple")
        return (float(value[0]), float(value[1]), float(value[2]))

    @model_validator(mode="after")
    def validate_action_fields(self) -> "ActionStep":
        if self.action == ActionType.MOVE_TO and self.target is None:
            raise ValueError("MOVE_TO requires target")
        if self.action == ActionType.GRIP and self.block is None:
            raise ValueError("GRIP requires block")
        if self.action == ActionType.RELEASE and (self.target is not None or self.block is not None):
            raise ValueError("RELEASE must not include target or block")
        return self


class ActionPlan(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    agent_name: str = "mock"
    actions: list[ActionStep] = Field(min_length=1)

    @classmethod
    def validate_raw_plan(cls, raw_plan: Any, agent_name: str = "mock") -> "ActionPlan":
        if isinstance(raw_plan, cls):
            return raw_plan
        if isinstance(raw_plan, list):
            return cls(agent_name=agent_name, actions=raw_plan)
        if isinstance(raw_plan, dict):
            return cls.model_validate(raw_plan)
        raise ValueError("action plan must be a list, dict, or ActionPlan")
