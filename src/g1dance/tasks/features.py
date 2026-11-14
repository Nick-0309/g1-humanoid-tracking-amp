"""Optional task variations (the tutorial's additions). Pure python so it can be saved next to a checkpoint."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace

FILE_NAME = "features.json"

LAYOUT_FIELDS = ("future_ref", "residual", "residual_scale", "slew_limit", "contact_obs")
"""Features that change the shape of the observations or the actions.

A checkpoint can only be loaded into a task with the same layout; the other features (reward terms, termination
thresholds) can be changed freely when initialising from a checkpoint."""


@dataclass(frozen=True)
class TaskFeatures:
    future_ref: tuple[int, ...] = ()
    """Also observe the reference joint pose and velocity this many control steps ahead (50 Hz: 5 = 0.1 s)."""
    residual: bool = False
    """Action is a residual on the reference pose of the next step instead of an offset from the default pose."""
    residual_scale: float | None = None
    """Radians per unit action for the residual; ``None`` keeps the upstream per-joint scale."""
    slew_limit: bool = False
    """Rate limit on the residual (arms 10 rad/s, legs/waist/ankles 20 rad/s). Needs ``residual``."""
    foot_rewards: bool = False
    """Add the landing bundle: reference-contact, swing-clearance, foot-height, anchor-height, feet-slide and
    action-magnitude reward terms (see :mod:`g1dance.mdp.rewards`)."""
    contact_obs: bool = False
    """Also observe the reference stance labels and the measured foot contact (2 + 2 numbers)."""
    foot_termination: float | None = None
    """End the episode when a foot's vertical error to the reference exceeds this many metres."""

    def __post_init__(self):
        if self.slew_limit and not self.residual:
            raise ValueError("slew_limit needs residual")
        if self.residual_scale is not None and not self.residual:
            raise ValueError("residual_scale needs residual")
        if any(o <= 0 for o in self.future_ref):
            raise ValueError("future_ref offsets must be positive")
        if self.foot_termination is not None and self.foot_termination <= 0:
            raise ValueError("foot_termination must be positive")

    @property
    def tag(self) -> str:
        """Short description for run names."""
        parts = []
        if self.future_ref:
            parts.append("fut" + "-".join(map(str, self.future_ref)))
        if self.residual:
            parts.append("res")
        if self.slew_limit:
            parts.append("slew")
        if self.foot_rewards:
            parts.append("foot")
        if self.contact_obs:
            parts.append("cobs")
        if self.foot_termination is not None:
            parts.append(f"fterm{self.foot_termination:g}")
        return "+".join(parts) or "base"

    @property
    def layout(self) -> dict:
        """The switches that the network input / output size depends on."""
        return {name: getattr(self, name) for name in LAYOUT_FIELDS}

    def with_layout_from(self, other: TaskFeatures) -> TaskFeatures:
        """These features, but with the observation / action layout of ``other``.

        Used by ``--init_from``: the actor and critic weights of the source checkpoint fix the layout, while the
        reward terms and the termination thresholds of *this* run can be new.
        """
        return replace(self, **other.layout)

    def save(self, run_dir: str) -> None:
        params = os.path.join(run_dir, "params")
        os.makedirs(params, exist_ok=True)
        with open(os.path.join(params, FILE_NAME), "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, run_dir: str) -> TaskFeatures | None:
        """Features of the run that wrote the checkpoints in ``run_dir``; ``None`` for runs without the file."""
        path = os.path.join(run_dir, "params", FILE_NAME)
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["future_ref"] = tuple(data.get("future_ref", ()))
        return cls(**data)


def parse_offsets(text: str | None) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",")) if text else ()
